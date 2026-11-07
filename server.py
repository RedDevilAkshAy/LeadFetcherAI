from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from pathlib import Path


from database import get_db, init_db
from schemas import (
    FindNew,
    LeadSchema,
    EditEmailSchema,
    LeadResponse,
    FullLeadReportResponse,
    AllLeadResponse
)
from lead_fetcher.models import Lead, BusinessAnalysis, LeadScore
from lead_fetcher.sourcer import LeadFinder, SourceData
from lead_fetcher.pipeline import process_lead_pipeline


BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


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



@app.post("/leads/{lead_id}/analyze", response_model=FullLeadReportResponse)
async def analyze_existing_lead(lead_id: int, db: Session = Depends(get_db)):
    """Trigger the full intelligence pipeline on demand for an existing lead."""
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    await process_lead_pipeline(lead_id=lead.id, db=db)
    db.refresh(lead)
    return SourceData().build_full_report(db=db, lead_id=lead.id)


@app.put("/leads/{lead_id}/email")
@app.patch("/leads/{lead_id}/email")
def update_lead_email(lead_id: int, payload: EditEmailSchema, db: Session = Depends(get_db)):
    """Update only the business owner's email for a specific lead."""
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    clean_email = payload.email.strip() if payload.email and payload.email.strip() else None

    # Check if duplicate email exists for another lead
    if clean_email:
        duplicate = db.query(Lead).filter(Lead.email == clean_email, Lead.id != lead_id).first()
        if duplicate:
            raise HTTPException(
                status_code=409,
                detail=f"Email '{clean_email}' is already associated with business '{duplicate.business_name}' (ID: {duplicate.id})",
            )

    lead.email = clean_email
    try:
        db.commit()
        db.refresh(lead)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Database constraint violation while saving email")

    return {
        "success": True,
        "lead_id": lead.id,
        "email": lead.email,
        "message": "Email updated successfully"
    }


@app.post("/leads/{lead_id}/send_to_crm")
async def send_lead_to_crm(lead_id: int, db: Session = Depends(get_db)):
    """Push an existing lead to Zoho CRM Leads module. Requires an email."""
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    email = (lead.email or lead.website_email or "").strip()
    if not email:
        raise HTTPException(
            status_code=400,
            detail="Cannot send to CRM: No email address available for this business."
        )

    analysis = lead.analyses[-1] if lead.analyses else None
    score = lead.score

    from lead_fetcher.connect_crm import push_lead_to_zoho
    return await push_lead_to_zoho(lead=lead, analysis=analysis, score=score)


@app.get("/get_leads", response_model=AllLeadResponse)
def get_leads(
    db: Session = Depends(get_db),
    owner_status: str = Query("all"),
    location: Optional[str] = Query(None),
    characteristic: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    result = SourceData().get_leads(
        db=db,
        owner_status=owner_status,
        location=location,
        characteristic=characteristic,
        limit=limit,
        offset=offset,
    )
    return result


@app.get("/api/dashboard_filters")
def get_dashboard_filters(db: Session = Depends(get_db)):
    """Fetch distinct locations and characteristics for dropdown filters."""
    return SourceData().dashboard_details()

@app.get("/leads/{lead_id}", response_model=FullLeadReportResponse)
def get_lead(lead_id: int, db: Session = Depends(get_db)):
    """Fetch a single lead by its ID with full intelligence:

    Website analysis, score, diagnosed opportunities, and generated WhatsApp/Email pitches.
    """
    return SourceData().build_full_report(db=db, lead_id=lead_id)


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

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/lead/{lead_id}", response_class=HTMLResponse)
async def dashboard(request: Request, lead_id: Optional[int] = None):
    dashboard_details = SourceData().dashboard_details()
    dashboard_details["initial_lead_id"] = lead_id
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=dashboard_details,
    )


# ---------------------------------------------------------------------------
# Zoho CRM OAuth Endpoints
# ---------------------------------------------------------------------------

@app.get("/zoho/status")
def zoho_status():
    """Return the current Zoho CRM connection status and the authorization URL."""
    from lead_fetcher.connect_crm import is_zoho_connected, get_zoho_auth_url
    return {
        "connected": is_zoho_connected(),
        "auth_url": get_zoho_auth_url(),
    }


@app.get("/zoho/connect")
def zoho_connect():
    """Redirect the browser to the Zoho OAuth consent page."""
    from lead_fetcher.connect_crm import get_zoho_auth_url
    return RedirectResponse(url=get_zoho_auth_url(), status_code=302)


@app.get("/zoho/callback")
async def zoho_callback(
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
):
    """Handle the Zoho OAuth redirect, exchange the code for tokens, persist the refresh token."""
    import logging as _logging
    from lead_fetcher.connect_crm import exchange_code_for_tokens, save_refresh_token

    if error:
        msg = error_description or error
        _logging.getLogger("server").error("Zoho OAuth callback error: %s", msg)
        return RedirectResponse(url=f"/dashboard?zoho=error&msg={msg}", status_code=302)

    if not code:
        return RedirectResponse(
            url="/dashboard?zoho=error&msg=No+authorization+code+received",
            status_code=302,
        )

    try:
        tokens = await exchange_code_for_tokens(code)
    except HTTPException as exc:
        return RedirectResponse(url=f"/dashboard?zoho=error&msg={exc.detail}", status_code=302)

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return RedirectResponse(
            url="/dashboard?zoho=error&msg=Zoho+did+not+return+a+refresh+token",
            status_code=302,
        )

    # Persist token to .env and update runtime config immediately
    save_refresh_token(refresh_token)
    return RedirectResponse(url="/dashboard?zoho=connected", status_code=302)
