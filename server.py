from __future__ import annotations

from typing import List
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db
from schemas import FindNew, LeadSchema, LeadResponse
from lead_fetcher.models import Lead
from lead_fetcher.sourcer import LeadFinder

app = FastAPI(
    title="Lead Fetcher AI",
    description="Automated business lead finder using Google Places API and Gemini API for owner extraction",
    version="1.0.0",
)


@app.post("/find_new")
async def find_new_leads(data: FindNew, db: Session = Depends(get_db)):
    """Fetch businesses for a given niche and locations, extract owner names via Gemini,

    and asynchronously save them to the database.
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
        "message": f"Successfully fetched and saved {len(leads_fetched)} leads.",
        "count": len(leads_fetched),
        "leads": leads_fetched,
    }


@app.get("/get_leads", response_model=List[LeadResponse])
def get_leads(db: Session = Depends(get_db)):
    """Fetch all leads saved in the database."""
    leads = db.query(Lead).all()
    return leads


@app.get("/leads/{lead_id}", response_model=LeadResponse)
def get_lead(lead_id: int, db: Session = Depends(get_db)):
    """Fetch a single lead by its ID."""
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead


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
