from __future__ import annotations

from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional, List


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


class LeadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    business_name: Optional[str] = None
    business_website: Optional[str] = None
    owner_name: Optional[str] = None
    email: Optional[str] = None
    number: Optional[str] = None
    characteristic: Optional[str] = None


class LeadSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: Optional[int] = None
    name: Optional[str] = None
    url: Optional[str] = None
    city: Optional[str] = None
    owner_name: Optional[str] = None
    business_type: Optional[str] = None
    number: Optional[str] = None
