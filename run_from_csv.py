"""
CSV-based Lead Ingestion & Intelligence Pipeline Runner.

Allows running the full LeadFetcher AI flow from a CSV file containing
business details (the same details normally fetched via Google Places API).

No owner details are expected in the CSV; the agent extracts the owner name
and executes the complete pipeline:

Flow per business in CSV:
1. Parse business details:
   - name / business_name
   - url / website
   - city / location
   - business_type / characteristic / niche
   - phone / number
   - rating & userRatingCount / review_count
   - business_profile (Google Maps URI / Places profile)
2. Domain duplicate check (skip if already in DB, unless --force).
3. Owner extraction: Gemini agent + Google Search grounding finds the owner's single name.
4. Database persistence: Saves the Lead record to DB with all details.
5. Full Intelligence Pipeline:
   - Website crawl & extract 'website_email'
   - Store pgvector RAG embeddings
   - Diagnose gaps with LangGraph agent
   - Compute opportunity score & priority
   - Generate personalized WhatsApp & Email pitches
   - Persist LeadScore and BusinessAnalysis in DB.

Usage::

    # Run using default sample CSV:
    python run_from_csv.py

    # Run using your custom CSV:
    python run_from_csv.py --csv path/to/businesses.csv

    # Force re-processing even if domain already exists in DB:
    python run_from_csv.py --csv businesses.csv --force
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from database import SessionLocal, init_db
from lead_fetcher.models import Lead
from lead_fetcher.pipeline import process_lead_pipeline
from lead_fetcher.sourcer import LeadFinder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_from_csv")


def _find_field(row: Dict[str, str], aliases: List[str]) -> Optional[str]:
    """Look up a value from a CSV row dictionary using multiple possible header aliases."""
    normalized_keys = {k.strip().lower().replace(" ", "_"): k for k in row.keys() if k}
    for alias in aliases:
        canonical = alias.strip().lower().replace(" ", "_")
        if canonical in normalized_keys:
            val = row[normalized_keys[canonical]]
            if val is not None and str(val).strip():
                return str(val).strip()
    return None


def parse_business_row(row: Dict[str, str], finder: LeadFinder) -> Optional[Dict[str, Any]]:
    """Extract business details matching the structure returned by Google Places API."""
    name = _find_field(row, ["name", "business_name", "company", "business", "company_name", "displayName"]) or "Unknown Business"
    url = _find_field(row, ["url", "website", "business_website", "websiteUri", "link", "site"]) or ""

    if not url:
        logger.warning(f"Skipping business '{name}': no website URL provided.")
        return None

    domain = finder.normalize_domain(url)
    if not domain:
        logger.warning(f"Skipping business '{name}': invalid website URL '{url}'.")
        return None

    city = _find_field(row, ["city", "location", "address", "locality", "state"]) or "Unknown"
    business_type = _find_field(row, ["business_type", "characteristic", "niche", "category", "googleMapsTypeLabel", "type"]) or ""
    phone = _find_field(row, ["phone", "number", "internationalPhoneNumber", "nationalPhoneNumber", "telephone", "mobile"]) or None
    business_profile = _find_field(row, ["business_profile", "googleMapsUri", "places_url", "maps_url", "profile_url", "profile"]) or None
    place_id = _find_field(row, ["place_id", "id"]) or None

    rating_raw = _find_field(row, ["rating", "stars", "score"])
    rating: Optional[float] = None
    if rating_raw:
        try:
            rating = float(rating_raw)
        except ValueError:
            pass

    reviews_raw = _find_field(row, ["userRatingCount", "review_count", "reviews", "ratings_count", "user_ratings"])
    user_ratings: Optional[int] = None
    if reviews_raw:
        try:
            user_ratings = int(float(reviews_raw))
        except ValueError:
            pass

    return {
        "place_id": place_id,
        "name": name,
        "url": url,
        "domain": domain,
        "city": city,
        "business_type": business_type,
        "phone": phone,
        "rating": rating,
        "userRatingCount": user_ratings,
        "business_profile": business_profile,
    }


async def process_csv(csv_path: str, force: bool = False, delay_seconds: float = 1.0) -> List[Dict[str, Any]]:
    """Reads business details from CSV, extracts owner, and executes the full intelligence pipeline."""
    file_path = Path(csv_path)
    if not file_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return []

    # Ensure database tables and columns exist
    try:
        init_db()
    except Exception as e:
        logger.warning(f"Note on init_db: {e}")

    finder = LeadFinder()
    results: List[Dict[str, Any]] = []

    # Read CSV rows
    rows: List[Dict[str, str]] = []
    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total = len(rows)
    logger.info(f"Loaded {total} business row(s) from {csv_path}")

    # Track already existing domains in database
    seen_domains = set()
    db = SessionLocal()
    try:
        if not force:
            existing_sites = db.query(Lead.business_website).all()
            for (site,) in existing_sites:
                d = finder.normalize_domain(site)
                if d:
                    seen_domains.add(d)
    finally:
        db.close()

    for idx, raw_row in enumerate(rows, start=1):
        business = parse_business_row(raw_row, finder)
        if not business:
            continue

        b_name = business["name"]
        b_url = business["url"]
        b_domain = business["domain"]
        b_city = business["city"]
        b_type = business["business_type"]

        logger.info(f"\n==================================================")
        logger.info(f"[{idx}/{total}] Sourcing business: '{b_name}' ({b_url})")

        if not force and b_domain in seen_domains:
            logger.info(f"Skipping '{b_name}': domain '{b_domain}' already in DB. Use --force to re-process.")
            continue

        seen_domains.add(b_domain)

        # ----------------------------------------------------
        # Flow Step 1: Owner extraction with Gemini API
        # ----------------------------------------------------
        logger.info(f"Fetching owner for '{b_name}' ({b_url}) via Gemini agent...")
        owner = await finder.get_owner_single_name(
            business_name=b_name,
            website_url=b_url,
            location=b_city,
        )
        logger.info(f"Owner identified for '{b_name}': '{owner}'")

        # ----------------------------------------------------
        # Flow Step 2: Save lead immediately to database
        # ----------------------------------------------------
        db = SessionLocal()
        lead_record = None
        try:
            lead_record = finder.save_lead_to_db(
                business=business,
                owner_name=owner,
                niche=b_type,
                db=db,
                location=b_city,
            )
        except Exception as e:
            logger.error(f"Error saving lead '{b_name}' to database: {e}")
            db.close()
            continue

        if not lead_record or not lead_record.id:
            logger.error(f"Failed to create database record for '{b_name}'.")
            db.close()
            continue

        lead_id = lead_record.id

        # ----------------------------------------------------
        # Flow Step 3: Run end-to-end intelligence pipeline
        # (Crawl & website_email -> pgvector RAG -> LangGraph diagnosis -> score -> pitch)
        # ----------------------------------------------------
        pipeline_output = None
        try:
            logger.info(f"Running intelligence pipeline for lead {lead_id}...")
            pipeline_output = await process_lead_pipeline(lead_id=lead_id, db=db)
            db.refresh(lead_record)
        except Exception as pe:
            logger.error(f"Error executing pipeline for lead {lead_id}: {pe}")
        finally:
            db.close()

        summary_entry = {
            "id": lead_id,
            "business_name": b_name,
            "business_website": b_url,
            "owner_name": lead_record.owner_name if lead_record else owner,
            "website_email": lead_record.website_email if lead_record else None,
            "business_profile": lead_record.business_profile if lead_record else None,
            "score": pipeline_output.get("score") if pipeline_output else None,
            "priority": pipeline_output.get("priority") if pipeline_output else "Unknown",
            "pitches_generated": bool(pipeline_output and pipeline_output.get("pitch_whatsapp")),
        }
        results.append(summary_entry)

        logger.info(
            f"Finished Lead {lead_id}: Score={summary_entry['score']}, "
            f"Priority={summary_entry['priority']}, "
            f"Website Email={summary_entry['website_email']}"
        )

        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    # ----------------------------------------------------
    # Summary Table
    # ----------------------------------------------------
    logger.info(f"\n==================================================")
    logger.info(f"CSV Pipeline Run Complete! Processed {len(results)}/{total} businesses.")
    logger.info(f"{'ID':<6} | {'Business Name':<30} | {'Owner':<20} | {'Website Email':<25} | {'Score':<6} | {'Priority'}")
    logger.info("-" * 105)
    for r in results:
        b_short = (r['business_name'][:28] + '..') if len(r['business_name']) > 28 else r['business_name']
        o_short = (r['owner_name'][:18] + '..') if r['owner_name'] and len(r['owner_name']) > 18 else (r['owner_name'] or 'Unknown')
        w_email = r['website_email'] or '-'
        sc = f"{r['score']:.0f}" if r['score'] is not None else '-'
        logger.info(f"{r['id']:<6} | {b_short:<30} | {o_short:<20} | {w_email:<25} | {sc:<6} | {r['priority']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the LeadFetcher AI pipeline from a CSV containing business details (Places API output format)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="sample_leads.csv",
        help="Path to CSV file with business details (default: sample_leads.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force ingestion and pipeline execution even if domain already exists in DB",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Polite delay in seconds between processing businesses (default: 1.0)",
    )

    args = parser.parse_args()
    asyncio.run(process_csv(csv_path=args.csv, force=args.force, delay_seconds=args.delay))


if __name__ == "__main__":
    main()
