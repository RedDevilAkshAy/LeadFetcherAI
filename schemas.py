from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr


class FindNew(BaseModel):
    niche: str
    locations: List[str]
    filter_ratings: Optional[bool] = False
    min_rating: Optional[float] = None
    min_reviews: Optional[int] = None
    max_reviews: Optional[int] = None


class LeadSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    business_name: Optional[str] = None
    business_website: Optional[str] = None
    owner_name: Optional[str] = None
    email: Optional[EmailStr] = None
    number: Optional[str] = None
    characteristic: Optional[str] = None
    location: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None


class LeadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    business_name: Optional[str] = None
    business_website: Optional[str] = None
    owner_name: Optional[str] = None
    email: Optional[str] = None
    number: Optional[str] = None
    characteristic: Optional[str] = None
    location: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    services_found: List[str] = []
    has_online_booking: bool = False
    has_whatsapp: bool = False
    has_contact_form: bool = False
    pricing_found: bool = False
    social_links: Dict[str, Optional[str]] = {}
    website_summary: Optional[str] = None


class ScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    score: Optional[float] = None
    priority: Optional[str] = None
    opportunities: List[str] = []
    recommended_actions: Optional[str] = None


class PitchResponse(BaseModel):
    whatsapp: Optional[str] = None
    email: Optional[str] = None


class FullLeadReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    business_name: Optional[str] = None
    location: Optional[str] = None
    business_website: Optional[str] = None
    owner_name: Optional[str] = None
    number: Optional[str] = None
    characteristic: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None

    analysis: Optional[AnalysisResponse] = None
    score: Optional[ScoreResponse] = None
    pitch: Optional[PitchResponse] = None

    formatted_card: Optional[str] = None
