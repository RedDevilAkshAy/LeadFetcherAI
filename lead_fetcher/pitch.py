from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict
from openai import OpenAI

try:
    from lead_fetcher.config import GEMINI_API_KEY, GEMINI_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL
except ImportError:
    from config import GEMINI_API_KEY, GEMINI_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL

logger = logging.getLogger("lead_fetcher.pitch")

_agent_nim_client = None

def get_agent_client():
    global _agent_nim_client

    if _agent_nim_client is None and NVIDIA_API_KEY:
        try:
            _agent_nim_client = OpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=NVIDIA_API_KEY,
            )
        except Exception:
            logger.exception("Failed to initialize NVIDIA NIM client")

    return _agent_nim_client


async def call_agent(prompt: str) -> str:
    """Call an NVIDIA NIM chat model with the supplied prompt."""
    client = get_agent_client()

    if client is None:
        logger.warning("NVIDIA NIM client is unavailable")
        return ""

    if not NVIDIA_MODEL:
        logger.warning("NVIDIA_MODEL is not configured")
        return ""

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=NVIDIA_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=8192,
        )

        if not response.choices:
            return ""

        return response.choices[0].message.content or ""

    except Exception:
        logger.exception("NVIDIA NIM call failed in agent")
        return ""


def _get_salutation(owner_name: str, niche: str) -> str:
    owner = (owner_name or "").strip()
    if not owner or owner.lower() == "unknown":
        if "dental" in niche.lower() or "clinic" in niche.lower():
            return "Hi Doctor,"
        return "Hi there,"

    if owner.lower().startswith("dr.") or owner.lower().startswith("dr "):
        return f"Hi {owner},"

    if "dental" in niche.lower() or "clinic" in niche.lower():
        return f"Hi Dr. {owner},"

    return f"Hi {owner},"


def _fallback_pitches(
    business_info: Dict[str, Any],
    analysis_data: Dict[str, Any],
    agent_result: Dict[str, Any],
) -> Dict[str, str]:
    name = business_info.get("business_name", "your business")
    owner = business_info.get("owner_name", "")
    niche = business_info.get("characteristic", "local business")
    reviews = business_info.get("review_count", 0) or 0
    rating = business_info.get("rating", 0.0) or 0.0

    salutation = _get_salutation(owner, niche)

    # Pick specific gaps
    gap_notes = []
    if reviews < 30:
        gap_notes.append(f"only {reviews} Google reviews")
    if not analysis_data.get("has_whatsapp", False):
        gap_notes.append("no direct WhatsApp booking option on the website")
    elif not analysis_data.get("has_online_booking", False):
        gap_notes.append("no instant online booking option")

    if not gap_notes:
        gap_notes.append("untapped local Google search visibility")

    gaps_str = " and ".join(gap_notes)

    whatsapp = (
        f"{salutation} I came across {name} on Google. I noticed your business has strong services "
        f"but {gaps_str}. I help local businesses improve Google visibility and turn searches into direct enquiries. "
        f"Would you be open to a free 10-minute audit this week?"
    )

    email = (
        f"Subject: Quick growth audit for {name}'s local visibility\n\n"
        f"{salutation}\n\n"
        f"I came across {name} on Google while reviewing top {niche} providers in your area. "
        f"You have an impressive reputation ({rating} stars), but I noticed a couple of friction points that might be costing you customers:\n\n"
        f"• {gaps_str}\n"
        f"• Opportunity to capture high-intent local searchers searching for {niche}\n\n"
        f"I help local businesses optimize their Google Business Profile and streamline website conversions into confirmed bookings.\n\n"
        f"Would you be open to a complimentary 10-minute audit this week to walk through 2-3 quick wins?\n\n"
        f"Best regards,\nLead Growth Team"
    )

    return {"whatsapp": whatsapp, "email": email}


async def generate_pitch(
    business_info: Dict[str, Any],
    analysis_data: Dict[str, Any],
    agent_result: Dict[str, Any],
    score_result: Dict[str, Any],
) -> Dict[str, str]:
    """Generates personalized WhatsApp and Email pitch messages using Gemini."""
    client = get_agent_client()
    if not client:
        return _fallback_pitches(business_info, analysis_data, agent_result)

    name = business_info.get("business_name", "the business")
    owner = business_info.get("owner_name", "Unknown")
    niche = business_info.get("characteristic", "local business")
    location = business_info.get("location", "")
    rating = business_info.get("rating", 0.0)
    reviews = business_info.get("review_count", 0)

    gaps = agent_result.get("gaps", [])
    opportunities = agent_result.get("opportunities", [])
    rec_actions = agent_result.get("recommended_actions", "")
    salutation = _get_salutation(owner, niche)

    prompt = f"""You are a top copywriter specialized in cold outreach for local B2B agency sales.
Generate two personalized pitches for this local business:

Business Name: {name}
Owner/Decision Maker: {owner}
Niche: {niche}
Location: {location}
Google Reviews: {rating} stars ({reviews} reviews)
Website Features:
- Online Booking: {analysis_data.get('has_online_booking', False)}
- WhatsApp Chat: {analysis_data.get('has_whatsapp', False)}
- Pricing: {analysis_data.get('pricing_found', False)}
Identified Gaps: {gaps}
Identified Opportunities: {opportunities}
Opportunity / Recommended Pitch: {rec_actions}
Priority: {score_result.get('priority', 'High')}

REQUIREMENTS:
1. "whatsapp": A punchy, natural WhatsApp message (approx 2-4 sentences, under 320 characters).
   - Start with appropriate greeting like "{salutation}".
   - Mention you came across {name} on Google.
   - Mention 1-2 specific observations (e.g. only {reviews} reviews and no direct WhatsApp booking).
   - Value statement on helping turn local searches into appointment enquiries.
   - Low-friction CTA: "Would you be open to a free 10-minute audit this week?"
2. "email": A professional cold email with Subject line and formatted body.
   - Specific, respectful, clear value proposition with actionable audit hook.

Respond ONLY with valid JSON in this exact structure:
{{
  "whatsapp": "...",
  "email": "..."
}}"""

    response = await call_agent(prompt)
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # fallback: extract JSON if the model adds text or Markdown fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        raw = fenced.group(1) if fenced else re.search(r"\{.*\}", response, re.DOTALL).group(0)
        data = json.loads(raw)

    whatsapp = data.get("whatsapp", "").strip()
    email = data.get("email", "").strip()

    if whatsapp and email:
        return {"whatsapp": whatsapp, "email": email}

    return _fallback_pitches(business_info, analysis_data, agent_result)
