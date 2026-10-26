from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from lead_fetcher.crawler import crawl_website
from lead_fetcher.models import BusinessAnalysis, BusinessEmbedding, Lead, LeadScore
from lead_fetcher.rag import chunk_text, store_embeddings
from lead_fetcher.agent import analyze_business
from lead_fetcher.scorer import score_lead
from lead_fetcher.pitch import generate_pitch

logger = logging.getLogger("lead_fetcher.pipeline")


async def process_lead_pipeline(lead_id: int, db: Session) -> Optional[Dict[str, Any]]:
    """Runs the complete intelligence pipeline for a saved Lead:

    1. Crawl website (services, booking, WhatsApp, pricing, forms)
    2. Save BusinessAnalysis to database
    3. Generate and store pgvector embeddings in BusinessEmbedding
    4. Execute LangGraph agent (RAG context -> diagnose gaps & opportunities)
    5. Calculate priority score (High / Medium / Low)
    6. Generate personalized WhatsApp and Email pitches
    7. Save LeadScore to database
    """
    lead = db.scalar(select(Lead).where(Lead.id == lead_id))
    if not lead:
        logger.error(f"Cannot process pipeline: Lead {lead_id} not found.")
        return None

    business_info = {
        "id": lead.id,
        "business_name": lead.business_name or "Unknown Business",
        "business_website": lead.business_website or "",
        "owner_name": lead.owner_name or "Unknown",
        "number": lead.number or "",
        "characteristic": lead.characteristic or "local business",
        "location": lead.location or "",
        "rating": lead.rating or 0.0,
        "review_count": lead.review_count or 0,
    }

    logger.info(f"--- [Pipeline Start] Processing Lead {lead.id}: '{lead.business_name}' ---")

    # 1. Crawl Website
    analysis_data = await crawl_website(lead.business_website or "")

    # 2. Save BusinessAnalysis in DB (upsert)
    existing_analysis = db.scalar(select(BusinessAnalysis).where(BusinessAnalysis.lead_id == lead.id))
    if not existing_analysis:
        existing_analysis = BusinessAnalysis(lead_id=lead.id)
        db.add(existing_analysis)

    existing_analysis.services_found = json.dumps(analysis_data.get("services_found", []))
    existing_analysis.has_online_booking = analysis_data.get("has_online_booking", False)
    existing_analysis.has_whatsapp = analysis_data.get("has_whatsapp", False)
    existing_analysis.has_contact_form = analysis_data.get("has_contact_form", False)
    existing_analysis.pricing_found = analysis_data.get("pricing_found", False)
    existing_analysis.social_links = json.dumps(analysis_data.get("social_links", {}))
    existing_analysis.website_summary = analysis_data.get("website_summary", "")
    existing_analysis.crawl_raw = analysis_data.get("crawl_text", "")

    try:
        db.commit()
        db.refresh(existing_analysis)
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save BusinessAnalysis for lead {lead.id}: {e}")

    # 3. Store pgvector RAG embeddings
    chunks_to_embed = []
    # Embed website summary
    if existing_analysis.website_summary:
        chunks_to_embed.append({
            "content_type": "summary",
            "text": f"{lead.business_name} in {lead.location}: {existing_analysis.website_summary}"
        })
    # Embed services
    services_list = analysis_data.get("services_found", [])
    if services_list:
        chunks_to_embed.append({
            "content_type": "services",
            "text": f"Services provided by {lead.business_name}: {', '.join(services_list)}"
        })
    # Embed crawl chunks
    raw_text = analysis_data.get("crawl_text", "")
    text_chunks = chunk_text(raw_text, chunk_size=300, overlap=30)
    for chunk in text_chunks[:6]:
        chunks_to_embed.append({
            "content_type": "crawl",
            "text": f"{lead.business_name} ({lead.location}): {chunk}"
        })

    await store_embeddings(lead_id=lead.id, content_chunks=chunks_to_embed, db=db)

    # 4. LangGraph Agent: diagnose gaps and growth opportunities
    agent_result = await analyze_business(
        business_info=business_info,
        analysis_data=analysis_data,
        db=db,
    )

    # 5. Lead Scoring (High / Medium / Low)
    score_result = score_lead(
        business_info=business_info,
        analysis_data=analysis_data,
        agent_result=agent_result,
    )

    # 6. Generate Pitch (WhatsApp & Email)
    pitch_result = await generate_pitch(
        business_info=business_info,
        analysis_data=analysis_data,
        agent_result=agent_result,
        score_result=score_result,
    )

    # 7. Save LeadScore to DB (upsert)
    existing_score = db.scalar(select(LeadScore).where(LeadScore.lead_id == lead.id))
    if not existing_score:
        existing_score = LeadScore(lead_id=lead.id)
        db.add(existing_score)

    existing_score.score = score_result.get("score", 0.0)
    existing_score.priority = score_result.get("priority", "Medium")
    existing_score.opportunities = json.dumps(agent_result.get("opportunities", []))
    existing_score.recommended_actions = agent_result.get("recommended_actions", "")
    existing_score.pitch_whatsapp = pitch_result.get("whatsapp", "")
    existing_score.pitch_email = pitch_result.get("email", "")

    try:
        db.commit()
        db.refresh(existing_score)
        logger.info(f"--- [Pipeline Complete] Lead {lead.id} scored {existing_score.score} ({existing_score.priority}) ---")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save LeadScore for lead {lead.id}: {e}")

    return {
        "lead_id": lead.id,
        "business_name": lead.business_name,
        "score": score_result.get("score"),
        "priority": score_result.get("priority"),
        "opportunities": agent_result.get("opportunities", []),
        "recommended_actions": agent_result.get("recommended_actions", ""),
        "pitch_whatsapp": pitch_result.get("whatsapp", ""),
        "pitch_email": pitch_result.get("email", ""),
    }
