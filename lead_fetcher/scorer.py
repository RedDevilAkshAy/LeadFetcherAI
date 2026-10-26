from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("lead_fetcher.scorer")


def score_lead(
    business_info: Dict[str, Any],
    analysis_data: Dict[str, Any],
    agent_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes a lead priority score (0-100) based on identified growth gaps and potential."""
    score = 0.0
    breakdown = {}

    reviews = business_info.get("review_count")
    if reviews is None:
        reviews = 0
    rating = business_info.get("rating")
    if rating is None:
        rating = 0.0

    # 1. Review Count Opportunity
    if reviews < 20:
        score += 20.0
        breakdown["low_reviews_under_20"] = 20.0
    elif reviews < 50:
        score += 10.0
        breakdown["moderate_reviews_under_50"] = 10.0

    # 2. Rating Opportunity (3.0 - 4.4 has good business with room for reputation boost)
    if 3.0 <= rating <= 4.4:
        score += 10.0
        breakdown["reputation_room_for_growth"] = 10.0

    # 3. Online Booking missing
    if not analysis_data.get("has_online_booking", False):
        score += 15.0
        breakdown["missing_online_booking"] = 15.0

    # 4. WhatsApp direct chat missing
    if not analysis_data.get("has_whatsapp", False):
        score += 15.0
        breakdown["missing_whatsapp"] = 15.0

    # 5. Contact form missing
    if not analysis_data.get("has_contact_form", False):
        score += 5.0
        breakdown["missing_contact_form"] = 5.0

    # 6. Pricing transparency missing
    if not analysis_data.get("pricing_found", False):
        score += 5.0
        breakdown["missing_pricing"] = 5.0

    # 7. Few / no structured services found
    services = analysis_data.get("services_found", [])
    if len(services) < 3:
        score += 10.0
        breakdown["sparse_service_pages"] = 10.0

    # 8. Missing socials
    socials = analysis_data.get("social_links", {})
    if isinstance(socials, dict):
        missing_count = sum(1 for v in socials.values() if not v)
        social_pts = min(10.0, missing_count * 2.0)
        if social_pts > 0:
            score += social_pts
            breakdown["missing_socials"] = social_pts

    # 9. Gaps identified by agent
    gaps = agent_result.get("gaps", [])
    gap_pts = min(10.0, len(gaps) * 2.0)
    if gap_pts > 0:
        score += gap_pts
        breakdown["agent_gaps_count"] = gap_pts

    # Cap at 100
    score = min(100.0, score)

    # Priority determination
    if score >= 65.0:
        priority = "High"
    elif score >= 40.0:
        priority = "Medium"
    else:
        priority = "Low"

    logger.info(f"Scored business '{business_info.get('business_name')}': {score} ({priority})")
    return {
        "score": round(score, 1),
        "priority": priority,
        "breakdown": breakdown,
    }
