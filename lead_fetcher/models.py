from __future__ import annotations

import json
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator

try:
    from database import Base, is_vector_available
except ImportError:
    from ..database import Base, is_vector_available


class VectorType(TypeDecorator):
    """Dynamically uses native pgvector VECTOR(dim) if available in PostgreSQL,

    otherwise falls back to Text storing a JSON serialized float array.
    This guarantees table creation works everywhere without requiring native C compilers on Windows.
    """
    impl = Text
    cache_ok = True

    def __init__(self, dim: int = 768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                if is_vector_available():
                    from pgvector.sqlalchemy import Vector
                    return dialect.type_descriptor(Vector(self.dim))
            except Exception:
                pass
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        try:
            if is_vector_available():
                return value
        except Exception:
            pass
        if isinstance(value, (list, tuple)):
            return json.dumps(value)
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value


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
    """Embeddings of business crawled content, services, and profile."""
    __tablename__ = "business_embeddings"

    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id", ondelete="CASCADE"), nullable=False)
    content_type = Column(String(50), nullable=False)     # "crawl", "services", "summary", etc.
    content_text = Column(Text, nullable=False)
    embedding = Column(VectorType(768), nullable=True)    # Native Vector or JSON-Text fallback
    created_at = Column(DateTime, server_default=func.now())

    lead = relationship("Lead", back_populates="embeddings")


class LeadScore(Base):
    """Agent-diagnosed score, opportunities, and personalized pitches."""
    __tablename__ = "lead_scores"

    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id", ondelete="CASCADE"), unique=True, nullable=False)
    score = Column(Float, nullable=True)                  # 0 to 100
    priority = Column(String(10), nullable=True)          # "High", "Medium", "Low"
    opportunities = Column(Text, nullable=True)           # JSON list of detected growth opportunities
    recommended_actions = Column(Text, nullable=True)     # Summary recommendation
    pitch_whatsapp = Column(Text, nullable=True)
    pitch_email = Column(Text, nullable=True)
    scored_at = Column(DateTime, server_default=func.now())

    lead = relationship("Lead", back_populates="score")
