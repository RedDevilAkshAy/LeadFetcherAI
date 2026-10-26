from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db, init_db
from schemas import (
    FindNew,
    LeadSchema,
    LeadResponse,
    FullLeadReportResponse,
    AnalysisResponse,
    ScoreResponse,
    PitchResponse,
)
from lead_fetcher.models import Lead, BusinessAnalysis, LeadScore
from lead_fetcher.sourcer import LeadFinder
from lead_fetcher.pipeline import process_lead_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    try:
        init_db()
    except Exception as e:
        import logging
        logging.getLogger("server").warning(f"Database initialization note: {e}")
    yield


app = FastAPI(
    title="Lead Fetcher AI",
    description="Automated local business lead intelligence engine: Google Places -> Crawler -> pgvector RAG -> LangGraph Agent -> Scorer -> Pitch Generator",
    version="2.0.0",
    lifespan=lifespan,
)


def format_lead_card(lead: Lead, analysis: Optional[BusinessAnalysis], score: Optional[LeadScore]) -> str:
    """Builds the exact human-readable diagnostic report card requested."""
    b_name = lead.business_name or "Unknown Business"
    loc = lead.location or ""
    biz_header = f"{b_name}, {loc}" if loc else b_name

    web_sum = analysis.website_summary if analysis and analysis.website_summary else (lead.business_website or "No website")
    rating_val = lead.rating or 0.0
    reviews_val = lead.review_count or 0
    reviews_line = f"{rating_val} stars, {reviews_val} reviews"

    opp_text = "Local visibility & conversion optimization"
    if score and score.opportunities:
        try:
            opp_list = json.loads(score.opportunities)
            if opp_list:
                opp_text = " + ".join(opp_list[:2])
        except Exception:
            opp_text = score.opportunities

    rec_pitch = score.recommended_actions if score and score.recommended_actions else "Improve review volume and add instant lead capture options"
    prio = score.priority if score and score.priority else "Medium"
    whatsapp_msg = score.pitch_whatsapp if score and score.pitch_whatsapp else "No pitch generated yet"

    return (
        f"Business: {biz_header}\n"
        f"Website: {web_sum}\n"
        f"Reviews: {reviews_line}\n"
        f"Opportunity: {opp_text}\n"
        f"Recommended pitch: {rec_pitch}\n"
        f"Priority: {prio}\n\n"
        f"Generated message:\n"
        f"{whatsapp_msg}"
    )


def build_full_report(lead: Lead) -> FullLeadReportResponse:
    analysis = lead.analyses[-1] if lead.analyses else None
    score = lead.score

    analysis_resp = None
    if analysis:
        services = []
        if analysis.services_found:
            try:
                services = json.loads(analysis.services_found)
            except Exception:
                services = [analysis.services_found]

        socials = {}
        if analysis.social_links:
            try:
                socials = json.loads(analysis.social_links)
            except Exception:
                socials = {}

        analysis_resp = AnalysisResponse(
            services_found=services,
            has_online_booking=analysis.has_online_booking,
            has_whatsapp=analysis.has_whatsapp,
            has_contact_form=analysis.has_contact_form,
            pricing_found=analysis.pricing_found,
            social_links=socials,
            website_summary=analysis.website_summary,
        )

    score_resp = None
    pitch_resp = None
    if score:
        opps = []
        if score.opportunities:
            try:
                opps = json.loads(score.opportunities)
            except Exception:
                opps = [score.opportunities]

        score_resp = ScoreResponse(
            score=score.score,
            priority=score.priority,
            opportunities=opps,
            recommended_actions=score.recommended_actions,
        )

        pitch_resp = PitchResponse(
            whatsapp=score.pitch_whatsapp,
            email=score.pitch_email,
        )

    formatted_card = format_lead_card(lead, analysis, score)

    return FullLeadReportResponse(
        id=lead.id,
        business_name=lead.business_name,
        location=lead.location,
        business_website=lead.business_website,
        owner_name=lead.owner_name,
        number=lead.number,
        characteristic=lead.characteristic,
        rating=lead.rating,
        review_count=lead.review_count,
        analysis=analysis_resp,
        score=score_resp,
        pitch=pitch_resp,
        formatted_card=formatted_card,
    )


@app.post("/find_new")
async def find_new_leads(data: FindNew, db: Session = Depends(get_db)):
    """Fetch businesses for any niche and locations via Google Places, extract owner names,

    crawl websites, store pgvector embeddings, diagnose with LangGraph, score priority,
    and generate personalized WhatsApp and email pitches.
    """
    finder = LeadFinder(niche=data.niche, locations=data.locations)

    leads_fetched = await finder.get_businesses(
        niche=data.niche,
        locations=data.locations,
        db=db,
        filter_ratings=data.filter_ratings or False,
        min_rating=data.min_rating,
        min_reviews=data.min_reviews,
        max_reviews=data.max_reviews,
    )

    return {
        "message": f"Successfully fetched, enriched, scored, and saved {len(leads_fetched)} leads.",
        "count": len(leads_fetched),
        "leads": leads_fetched,
    }


@app.get("/leads/{lead_id}", response_model=FullLeadReportResponse)
def get_lead(lead_id: int, db: Session = Depends(get_db)):
    """Fetch a single lead by its ID with full intelligence:

    Website analysis, score, diagnosed opportunities, and generated WhatsApp/Email pitches.
    """
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    return build_full_report(lead)


@app.post("/leads/{lead_id}/analyze", response_model=FullLeadReportResponse)
async def analyze_existing_lead(lead_id: int, db: Session = Depends(get_db)):
    """Trigger the full intelligence pipeline on demand for an existing lead."""
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    await process_lead_pipeline(lead_id=lead.id, db=db)
    db.refresh(lead)
    return build_full_report(lead)


@app.get("/get_leads", response_model=List[LeadResponse])
def get_leads(db: Session = Depends(get_db)):
    """Fetch all leads saved in the database."""
    leads = db.query(Lead).all()
    return leads


@app.post("/clean_leads")
def clean_existing_leads(db: Session = Depends(get_db)):
    """Clean all existing owner_name values in the database from full sentences to single full names."""
    finder = LeadFinder()
    leads = db.query(Lead).all()
    updated_records = []

    for lead in leads:
        old_name = lead.owner_name
        if not old_name or old_name == "Unknown":
            continue

        cleaned = finder.clean_owner_name(old_name, lead.business_name or "")
        if cleaned != old_name:
            lead.owner_name = cleaned
            updated_records.append(
                {
                    "id": lead.id,
                    "business_name": lead.business_name,
                    "before": old_name,
                    "after": cleaned,
                }
            )

    if updated_records:
        db.commit()

    return {
        "message": f"Successfully cleaned {len(updated_records)} leads.",
        "updated_count": len(updated_records),
        "changes": updated_records,
    }


@app.post("/create", response_model=LeadResponse)
def create_lead(lead: LeadSchema, db: Session = Depends(get_db)):
    """Manually add a new lead."""
    if lead.email:
        existing_lead = db.scalar(
            select(Lead).where(Lead.email == str(lead.email))
        )
        if existing_lead:
            raise HTTPException(
                status_code=409,
                detail="Lead with this email already exists",
            )

    new_lead = Lead(**lead.model_dump(exclude_unset=True))
    db.add(new_lead)
    try:
        db.commit()
        db.refresh(new_lead)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation")

    return new_lead


@app.put("/edit_lead/{lead_id}", response_model=LeadResponse)
def edit_leads(lead_id: int, lead: LeadSchema, db: Session = Depends(get_db)):
    """Update an existing lead by ID."""
    fetch_lead = db.query(Lead).filter_by(id=lead_id).first()
    if not fetch_lead:
        raise HTTPException(status_code=404, detail="Lead doesn't exist")

    for key, value in lead.model_dump(exclude_unset=True).items():
        setattr(fetch_lead, key, value)

    try:
        db.commit()
        db.refresh(fetch_lead)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Duplicate value or database constraint violation",
        )

    return fetch_lead


@app.delete("/delete/{lead_id}")
def delete_lead(lead_id: int, db: Session = Depends(get_db)):
    """Delete a lead by ID."""
    fetch_lead = db.query(Lead).filter_by(id=lead_id).first()
    if not fetch_lead:
        raise HTTPException(status_code=404, detail="Lead doesn't exist")

    db.delete(fetch_lead)
    db.commit()
    return {"message": f"Lead {lead_id} deleted successfully"}
