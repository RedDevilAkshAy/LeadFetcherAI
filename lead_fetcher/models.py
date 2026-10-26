from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for environments or tests where pgvector is not yet installed
    from sqlalchemy.types import TypeDecorator
    class Vector(TypeDecorator):
        impl = Text
        cache_ok = True

try:
    from database import Base
except ImportError:
    from ..database import Base


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True)
    business_name = Column(String(100), nullable=True)
    business_website = Column(String(100), nullable=True)
    owner_name = Column(String(50), nullable=True)
    email = Column(String(255), unique=True, nullable=True)
    number = Column(String(20), nullable=True)
    characteristic = Column(String(50), nullable=True)
    location = Column(String(100), nullable=True)
    rating = Column(Float, nullable=True)
    review_count = Column(Integer, nullable=True)

    analyses = relationship("BusinessAnalysis", back_populates="lead", cascade="all, delete-orphan")
    embeddings = relationship("BusinessEmbedding", back_populates="lead", cascade="all, delete-orphan")
    score = relationship("LeadScore", back_populates="lead", uselist=False, cascade="all, delete-orphan")


class BusinessAnalysis(Base):
    """Stores crawled website intelligence for any local business."""
    __tablename__ = "business_analyses"

    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id", ondelete="CASCADE"), nullable=False)
    services_found = Column(Text, nullable=True)          # JSON list of services detected
    has_online_booking = Column(Boolean, default=False)
    has_whatsapp = Column(Boolean, default=False)
    has_contact_form = Column(Boolean, default=False)
    pricing_found = Column(Boolean, default=False)
    social_links = Column(Text, nullable=True)            # JSON dict {platform: url}
    website_summary = Column(Text, nullable=True)
    crawl_raw = Column(Text, nullable=True)               # Combined crawl text for RAG
    crawled_at = Column(DateTime, server_default=func.now())

    lead = relationship("Lead", back_populates="analyses")


class BusinessEmbedding(Base):
    """pgvector embeddings of business crawled content, services, and profile."""
    __tablename__ = "business_embeddings"

    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id", ondelete="CASCADE"), nullable=False)
    content_type = Column(String(50), nullable=False)     # "crawl", "services", "summary", etc.
    content_text = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=True)        # 768-dim vector from Gemini embedding
    created_at = Column(DateTime, server_default=func.now())

    lead = relationship("Lead", back_populates="embeddings")


class LeadScore(Base):
    """Agent-diagnosed score, opportunities, and personalized pitches."""
    __tablename__ = "lead_scores"

    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id", ondelete="CASCADE"), unique=True, nullable=False)
    score = Column(Float, nullable=True)                  # 0 to 100
    priority = Column(String(10), nullable=True)           # "high", "medium", "low"
    opportunities = Column(Text, nullable=True)           # JSON list of detected growth opportunities
    recommended_actions = Column(Text, nullable=True)     # Summary recommendation
    pitch_whatsapp = Column(Text, nullable=True)
    pitch_email = Column(Text, nullable=True)
    scored_at = Column(DateTime, server_default=func.now())

    lead = relationship("Lead", back_populates="score")
