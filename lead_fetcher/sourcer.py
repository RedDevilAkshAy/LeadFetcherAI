from __future__ import annotations

import asyncio
import logging
from pprint import pprint
from typing import List, Optional, Union
from urllib.parse import urlparse

import httpx
from google import genai
from google.genai import types
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

try:
    from lead_fetcher.config import (
        GEMINI_API_KEY,
        GEMINI_MODEL,
        GOOGLE_API_KEY,
        GOOGLE_URL,
    )
    from lead_fetcher.models import Lead
    from lead_fetcher.pipeline import process_lead_pipeline
except ImportError:
    from config import (
        GEMINI_API_KEY,
        GEMINI_MODEL,
        GOOGLE_API_KEY,
        GOOGLE_URL,
    )
    from models import Lead
    try:
        from pipeline import process_lead_pipeline
    except ImportError:
        process_lead_pipeline = None

try:
    from database import SessionLocal, get_db
except ImportError:
    from ..database import SessionLocal, get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lead_fetcher.sourcer")


class LeadFinder:
    def __init__(
        self,
        niche: str = "",
        locations: Optional[Union[str, List[str]]] = None,
        google_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        google_url: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ):
        self.niche = niche
        self.locations = locations or []
        self.google_api_key = google_api_key or GOOGLE_API_KEY
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY
        self.google_url = (google_url or GOOGLE_URL or "https://places.googleapis.com/v1/places").rstrip("/")
        self.gemini_model = gemini_model or GEMINI_MODEL or "gemini-2.5-flash"

        self.google_client = None
        if self.gemini_api_key:
            try:
                self.google_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

    # -------------------------
    # DOMAIN NORMALIZATION
    # -------------------------
    def normalize_domain(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None

        url_str = str(url).strip().lower()
        if not url_str.startswith("http"):
            url_str = "https://" + url_str

        try:
            domain = urlparse(url_str).netloc
            domain = domain.replace("www.", "")
            return domain.split(":")[0].rstrip(".")
        except Exception:
            return None

    # -------------------------
    # GOOGLE PLACES SEARCH
    # -------------------------
    async def get_places(self, niche: str, location: str, client: httpx.AsyncClient) -> List[str]:
        url = f"{self.google_url}:searchText"

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.google_api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,nextPageToken",
        }

        place_ids = []
        seen_ids = set()
        next_token = None

        while True:
            payload = {
                "textQuery": f"{niche} {location}",
                "pageSize": 20,
            }

            if next_token:
                payload["pageToken"] = next_token
                await asyncio.sleep(2)

            try:
                resp = await client.post(url, headers=headers, json=payload, timeout=30.0)
                if resp.status_code != 200:
                    logger.error(
                        f"Places search failed with status {resp.status_code}: {resp.text}"
                    )
                    break

                data = resp.json()
                places = data.get("places", [])
                if not places:
                    break

                for p in places:
                    pid = p.get("id")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        place_ids.append(pid)

                next_token = data.get("nextPageToken")
                if not next_token:
                    break

            except Exception as e:
                logger.error(f"Error querying places for '{niche} {location}': {e}")
                break

        logger.info(f"Found {len(place_ids)} place IDs for '{niche}' in '{location}'")
        return place_ids

    # -------------------------
    # BUSINESS DETAILS FETCH
    # -------------------------
    async def get_business_details(
        self,
        place_id: str,
        client: httpx.AsyncClient,
        filter_ratings: bool = False,
        min_rating: Optional[float] = None,
        min_reviews: Optional[int] = None,
        max_reviews: Optional[int] = None,
    ) -> Optional[dict]:
        url = f"{self.google_url}/{place_id}"

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.google_api_key,
            "X-Goog-FieldMask": "displayName,websiteUri,addressComponents,googleMapsTypeLabel,rating,userRatingCount,googleMapsUri,internationalPhoneNumber,nationalPhoneNumber",
        }

        try:
            resp = await client.get(url, headers=headers, timeout=30.0)
            if resp.status_code != 200:
                logger.warning(
                    f"Failed to fetch place details for {place_id} (status {resp.status_code})"
                )
                return None

            data = resp.json()

            website = data.get("websiteUri")
            if not website:
                return None

            rating = data.get("rating", 0)
            user_ratings = data.get("userRatingCount", 0)

            # Optional rating / review count filter
            if filter_ratings:
                _min_rating = min_rating if min_rating is not None else 4.0
                _min_reviews = min_reviews if min_reviews is not None else 20
                _max_reviews = max_reviews if max_reviews is not None else 180

                if user_ratings <= _min_reviews or user_ratings >= _max_reviews or rating < _min_rating:
                    return None

            name = data.get("displayName", {}).get("text", "Unknown")
            domain = self.normalize_domain(website)
            if not domain:
                return None

            city = "Unknown"
            for c in data.get("addressComponents", []):
                if "locality" in c.get("types", []):
                    city = c.get("longText", "Unknown")
                    break

            phone = data.get("internationalPhoneNumber") or data.get("nationalPhoneNumber")
            business_type = data.get("googleMapsTypeLabel", {}).get("text", "")

            return {
                "place_id": place_id,
                "name": name,
                "url": website,
                "domain": domain,
                "city": city,
                "business_type": business_type,
                "phone": phone,
                "rating": rating,
                "userRatingCount": user_ratings,
            }

        except Exception as e:
            logger.error(f"Error getting business details for {place_id}: {e}")
            return None

    # -------------------------
    # SANITIZE & EXTRACT OWNER NAME
    # -------------------------
    def clean_owner_name(self, raw_text: str, business_name: str = "") -> str:
        """Extract a clean, single full name from Gemini output or existing database text

        without making any additional API calls.
        """
        if not raw_text:
            return "Unknown"

        import re

        text = raw_text.strip().strip('*_`"\'')
        if not text or "unknown" in text.lower():
            return "Unknown"

        # Remove citations like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text).strip()

        # Remove prefix keywords (e.g. "Owner: John Doe", "Founder: Jane Smith")
        text = re.sub(
            r"^(?:owner|founder|ceo|director|in charge|name)\s*:\s*",
            "",
            text,
            flags=re.I,
        ).strip()

        # Check for negative / non-informative statements
        negatives = [
            "not found",
            "could not find",
            "not disclose",
            "not available",
            "no information",
            "cannot find",
            "unable to find",
            "unclear",
            "n/a",
            "none",
        ]
        if any(neg in text.lower() for neg in negatives):
            return "Unknown"

        # Pattern 1: '<Business> was/is (co-)?founded / owned / created / established / started by <Name>...'
        # or 'Founded by <Name>...'
        m1 = re.search(
            r"\b(?:(?:was|is)\s+)?(?:co-)?(?:founded|owned|created|established|started|operated|run|led|opened)(?:\s+and\s+(?:operated|run|managed|owned))?\s+by\s+(.+)",
            text,
            flags=re.I,
        )
        if m1:
            after_by = m1.group(1).strip()
            after_by = re.split(
                r"\s+\b(?:in\s+\d{4}|in\s+[A-Z][a-z]+|who\b|which\b|and\s+(?:serves|is|acts|oversees)|alongside|with\b|as\s+well|to\b)\b|[,;\n](?!\s*(?:MD|DO|DNP|APRN|RN|PA-C|PhD|MBA|DDS|DMD|Jr|Sr))",
                after_by,
                flags=re.I,
            )[0].strip()
            after_by = re.sub(
                r"(?<!\bDr)(?<!\bMr)(?<!\bMs)[,\.\s]+$", "", after_by
            ).strip()
            if after_by and len(after_by.split()) <= 6:
                return after_by[:50]

        # Pattern 2: '(The )?(owner|founder|ceo|director) of <Business> is <Name>...'
        m2 = re.search(
            r"\b(?:the\s+)?(?:owner|founder|ceo|director|person in charge|medical director)(?:\s+(?:and|&)\s+(?:founder|owner|ceo|director))?(?:\s+of\s+[^,]+)?\s+(?:is|was)\s+(.+)",
            text,
            flags=re.I,
        )
        if m2:
            after_is = m2.group(1).strip()
            after_is = re.split(
                r"\s+\b(?:who\b|which\b|and\s+(?:serves|is|acts)|in\s+\d{4})\b|[,;\n](?!\s*(?:MD|DO|DNP|APRN|RN|PA-C|PhD|MBA|DDS|DMD|Jr|Sr))",
                after_is,
                flags=re.I,
            )[0].strip()
            after_is = re.sub(
                r"(?<!\bDr)(?<!\bMr)(?<!\bMs)[,\.\s]+$", "", after_is
            ).strip()
            if after_is and len(after_is.split()) <= 6:
                return after_is[:50]

        # Pattern 3: '<Name> is/serves as the (founder|owner|ceo)...' or '<Name> (founded|owns|operates)...'
        m3 = re.match(
            r"^(.+?)\s+(?:(?:is|serves as|acts as)\s+(?:the\s+|an?\s+)?(?:co-)?(?:founder|owner|ceo|director|president|creator|medical director|proprietor)|(?:founded|co-founded|owns|operates|runs|started))\b",
            text,
            flags=re.I,
        )
        if m3:
            before_verb = m3.group(1).strip()
            before_verb = re.sub(
                r"(?<!\bDr)(?<!\bMr)(?<!\bMs)[,\.\s]+$", "", before_verb
            ).strip()
            if before_verb and len(before_verb.split()) <= 6:
                return before_verb[:50]

        # Pattern 4: '<Name>, (the )?(founder|owner|ceo)...'
        m4 = re.match(
            r"^(.+?),\s*(?:the\s+|an?\s+)?(?:co-)?(?:founder|owner|ceo|director|president|medical director)\b",
            text,
            flags=re.I,
        )
        if m4:
            before_title = m4.group(1).strip()
            before_title = re.sub(
                r"(?<!\bDr)(?<!\bMr)(?<!\bMs)[,\.\s]+$", "", before_title
            ).strip()
            if before_title and len(before_title.split()) <= 6:
                return before_title[:50]

        # Pattern 5: Already clean name, e.g. "Ava Franzoni", "Justin Jeter, MD"
        cleaned = re.sub(r"(?<!\bDr)(?<!\bMr)(?<!\bMs)\.+$", "", text).strip()
        words = cleaned.split()
        if len(words) <= 6 and not any(
            w.lower()
            in {
                "is",
                "was",
                "are",
                "were",
                "the",
                "of",
                "for",
                "founded",
                "owns",
                "operates",
            }
            for w in words
        ):
            return cleaned[:50].strip()

        # Pattern 6: Fallback - take first sentence/clause
        first_clause = re.split(
            r"[,;\n](?!\s*(?:MD|DO|DNP|APRN|RN|PA-C|PhD|MBA|DDS|DMD))|(?<!\bDr)(?<!\bMr)(?<!\bMs)\.",
            text,
        )[0].strip()
        if 1 <= len(first_clause.split()) <= 5:
            return first_clause[:50].strip()

        return "Unknown"

    def sanitize_owner_name(self, raw_text: str) -> str:
        """Alias for clean_owner_name for backward compatibility."""
        return self.clean_owner_name(raw_text)

    # -------------------------
    # OWNER EXTRACTION WITH GEMINI
    # -------------------------
    async def get_owner_single_name(
        self, business_name: str, website_url: str, location: str = ""
    ) -> str:
        if not self.gemini_api_key or not self.google_client:
            logger.warning("GEMINI_API_KEY is not set. Defaulting owner to 'Unknown'.")
            return "Unknown"

        prompt = f"""Your task for this session: I'm going to paste the business name of {self.niche or "the business"} in {location or self.locations}.
Fetch the full name of the owner, founder, or person in charge.
The format provided: {business_name} - {website_url}

CRITICAL RULES:
1. Return ONLY the person's single Full Name (e.g. "Jane Doe" or "Dr. John Smith, MD").
2. Do NOT write full sentences (do NOT say "Jane Doe is the founder of..." or "Founded by...").
3. Do NOT include explanations, job titles, or prefixes like "Owner:".
4. If you cannot identify the person with confidence, return exactly "Unknown".

Here we go: {business_name} - {website_url}"""

        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
        )

        try:
            # Use native async if supported by google-genai
            if hasattr(self.google_client, "aio") and hasattr(
                self.google_client.aio, "models"
            ):
                response = await self.google_client.aio.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=config,
                )
            else:
                response = await asyncio.to_thread(
                    self.google_client.models.generate_content,
                    model=self.gemini_model,
                    contents=prompt,
                    config=config,
                )

            raw_text = response.text if response and response.text else "Unknown"
            return self.clean_owner_name(raw_text, business_name)

        except Exception as e:
            err_msg = str(e)
            logger.warning(
                f"Gemini query for '{business_name}' with model '{self.gemini_model}' failed: {err_msg}"
            )

            # Fallback to gemini-2.0-flash if specified model is not found
            if (
                "not found" in err_msg.lower() or "not supported" in err_msg.lower()
            ) and self.gemini_model != "gemini-2.0-flash":
                try:
                    logger.info(
                        f"Retrying owner lookup with fallback model 'gemini-2.0-flash' for '{business_name}'..."
                    )
                    if hasattr(self.google_client, "aio") and hasattr(
                        self.google_client.aio, "models"
                    ):
                        response = await self.google_client.aio.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=prompt,
                            config=config,
                        )
                    else:
                        response = await asyncio.to_thread(
                            self.google_client.models.generate_content,
                            model="gemini-2.0-flash",
                            contents=prompt,
                            config=config,
                        )

                    raw_text = response.text if response and response.text else "Unknown"
                    return self.clean_owner_name(raw_text, business_name)

                except Exception as fb_err:
                    logger.error(f"Fallback model also failed: {fb_err}")
                    return "Unknown"

            return "Unknown"

    # -------------------------
    # DATABASE PERSISTENCE
    # -------------------------
    def save_lead_to_db(
        self,
        business: dict,
        owner_name: str,
        niche: str,
        db: Session,
        location: str = "",
    ) -> Optional[Lead]:
        try:
            phone = (
                business.get("phone")
                or business.get("internationalPhoneNumber")
                or business.get("nationalPhoneNumber")
            )
            if phone:
                phone = str(phone).strip()[:20]

            char_val = business.get("business_type") or niche or ""
            char_val = str(char_val).strip()[:20]

            name_val = str(business.get("name") or "Unknown").strip()[:100]
            url_val = str(business.get("url") or "").strip()[:100]
            owner_val = str(owner_name or "Unknown").strip()[:50]

            lead = Lead(
                business_name=name_val,
                business_website=url_val,
                owner_name=owner_val,
                email=None,  # None avoids unique constraint collision
                number=phone if phone else None,
                characteristic=char_val if char_val else None,
                location=str(location).strip()[:100] if location else None,
                rating=business.get("rating"),
                review_count=business.get("userRatingCount"),
            )

            db.add(lead)
            db.commit()
            db.refresh(lead)
            logger.info(
                f"Saved lead to DB [ID: {lead.id}] - '{lead.business_name}' (Owner: '{lead.owner_name}')"
            )
            return lead

        except IntegrityError as ie:
            db.rollback()
            logger.warning(f"Database IntegrityError saving lead '{business.get('name')}': {ie}")
            return None
        except Exception as e:
            db.rollback()
            logger.error(f"Database error saving lead '{business.get('name')}': {e}")
            return None

    # -------------------------
    # MAIN WORKFLOW
    # -------------------------
    async def get_businesses(
        self,
        niche: Optional[str] = None,
        locations: Optional[Union[str, List[str]]] = None,
        db: Optional[Session] = None,
        filter_ratings: bool = False,
        min_rating: Optional[float] = None,
        min_reviews: Optional[int] = None,
        max_reviews: Optional[int] = None,
    ) -> List[dict]:
        niche = niche or self.niche
        locations = locations or self.locations

        if isinstance(locations, str):
            locations = [locations]
        elif not locations:
            locations = []

        self.niche = niche
        self.locations = locations

        close_db_on_exit = False
        if db is None:
            db = SessionLocal()
            close_db_on_exit = True

        saved_leads = []
        seen_domains = set()

        try:
            # Pre-load existing domains from database to prevent duplicate processing
            try:
                existing_sites = db.query(Lead.business_website).all()
                for (site,) in existing_sites:
                    d = self.normalize_domain(site)
                    if d:
                        seen_domains.add(d)
            except Exception as e:
                logger.warning(f"Could not load existing domains from DB: {e}")

            async with httpx.AsyncClient() as client:
                for location in locations:
                    logger.info(f"Scraping niche '{niche}' in '{location}'...")
                    place_ids = await self.get_places(niche, location, client)

                    # Iterate through them one by one asynchronously
                    for place_id in place_ids:
                        business = await self.get_business_details(
                            place_id=place_id,
                            client=client,
                            filter_ratings=filter_ratings,
                            min_rating=min_rating,
                            min_reviews=min_reviews,
                            max_reviews=max_reviews,
                        )

                        if not business:
                            continue

                        domain = business.get("domain")
                        if domain in seen_domains:
                            logger.info(f"Skipping already seen domain: {domain}")
                            continue

                        seen_domains.add(domain)

                        # Owner extraction with Gemini API
                        logger.info(
                            f"Fetching owner for '{business['name']}' ({business['url']})..."
                        )
                        owner = await self.get_owner_single_name(
                            business_name=business["name"],
                            website_url=business["url"],
                            location=location,
                        )

                        # Save lead immediately to database
                        lead_record = self.save_lead_to_db(
                            business=business,
                            owner_name=owner,
                            niche=niche,
                            db=db,
                            location=location
                        )

                        # Run end-to-end intelligence pipeline: crawl -> pgvector -> agent -> score -> pitch
                        if lead_record and lead_record.id and process_lead_pipeline:
                            try:
                                logger.info(f"Running intelligence pipeline for lead {lead_record.id}...")
                                await process_lead_pipeline(lead_id=lead_record.id, db=db)
                            except Exception as pe:
                                logger.error(f"Error executing pipeline for lead {lead_record.id}: {pe}")

                        lead_data = {
                            "id": lead_record.id if lead_record else None,
                            "name": business["name"],
                            "url": business["url"],
                            "domain": domain,
                            "city": business["city"],
                            "owner_name": owner,
                            "number": business.get("phone"),
                            "business_type": business.get("business_type"),
                            "location": location,
                        }
                        saved_leads.append(lead_data)

                        # Short delay between items to avoid hitting rate limits
                        await asyncio.sleep(1)

                    logger.info(
                        f"Finished location '{location}'. {len(saved_leads)} total leads accumulated."
                    )

        finally:
            if close_db_on_exit:
                db.close()

        return saved_leads

    def run(
        self,
        niche: Optional[str] = None,
        locations: Optional[Union[str, List[str]]] = None,
        filter_ratings: bool = False,
        min_rating: Optional[float] = None,
        min_reviews: Optional[int] = None,
        max_reviews: Optional[int] = None,
    ) -> List[dict]:
        """Synchronous wrapper around get_businesses."""
        return asyncio.run(
            self.get_businesses(
                niche=niche,
                locations=locations,
                filter_ratings=filter_ratings,
                min_rating=min_rating,
                min_reviews=min_reviews,
                max_reviews=max_reviews,
            )
        )


if __name__ == "__main__":
    locations_list = [
        "Dubai, UAE",
        "Abu Dhabi, UAE",
        "Sharjah, UAE",
    ]
    finder = LeadFinder(
        niche="Aesthetic clinics",
        locations=locations_list,
    )
    results = finder.run()
    print(f"Finished scraping. Total leads saved: {len(results)}")
    pprint(results)
