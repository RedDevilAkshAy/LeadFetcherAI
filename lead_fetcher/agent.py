from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, START, END

try:
    from lead_fetcher.config import GEMINI_API_KEY, GEMINI_MODEL
    from lead_fetcher.rag import retrieve_similar
except ImportError:
    from config import GEMINI_API_KEY, GEMINI_MODEL
    from rag import retrieve_similar

logger = logging.getLogger("lead_fetcher.agent")

_agent_genai_client = None


def get_agent_client():
    global _agent_genai_client
    if _agent_genai_client is None and GEMINI_API_KEY:
        try:
            from google import genai
            _agent_genai_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client in agent: {e}")
    return _agent_genai_client


async def call_gemini(prompt: str) -> str:
    """Helper to call Gemini LLM with prompt."""
    client = get_agent_client()
    if not client:
        return ""

    model = GEMINI_MODEL or "gemini-2.5-flash"
    try:
        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
            )
        else:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=prompt,
            )
        return response.text if response and response.text else ""
    except Exception as e:
        logger.warning(f"Gemini call failed in agent: {e}")
        return ""


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    business_info: Dict[str, Any]
    analysis_data: Dict[str, Any]
    rag_context: str
    web_presence_assessment: str
    review_assessment: str
    gaps: List[str]
    opportunities: List[str]
    recommended_actions: str


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

async def analyze_web_presence(state: AgentState) -> Dict[str, Any]:
    """Node 1: Assess website presence, booking capability, WhatsApp, and service depth."""
    b_info = state["business_info"]
    a_data = state["analysis_data"]

    name = b_info.get("business_name", "Unknown Business")
    website = b_info.get("business_website", "")
    niche = b_info.get("characteristic", "local business")

    has_booking = a_data.get("has_online_booking", False)
    has_whatsapp = a_data.get("has_whatsapp", False)
    has_form = a_data.get("has_contact_form", False)
    services = a_data.get("services_found", [])
    pricing = a_data.get("pricing_found", False)

    prompt = f"""You are a senior digital growth strategist for local businesses.
Analyze the online presence for this business:
Business: {name}
Niche: {niche}
Website: {website}
Online Booking Available: {has_booking}
WhatsApp Direct Button: {has_whatsapp}
Contact Form: {has_form}
Pricing/Rates Listed: {pricing}
Services Discovered ({len(services)}): {', '.join(services[:10]) if services else 'None clearly found'}

Write a concise 2-sentence diagnostic assessment of this business's website conversion capabilities.
Focus on whether high-intent visitors can easily convert (e.g. book or WhatsApp) or if friction exists."""

    assessment = await call_gemini(prompt)
    if not assessment:
        missing = []
        if not has_booking:
            missing.append("online booking")
        if not has_whatsapp:
            missing.append("direct WhatsApp enquiry")
        if not services:
            missing.append("dedicated service pages")
        assessment = (
            f"Website present but suffers from lack of {', '.join(missing)}."
            if missing
            else "Website has standard conversion assets available."
        )

    return {"web_presence_assessment": assessment.strip()}


async def analyze_reviews(state: AgentState) -> Dict[str, Any]:
    """Node 2: Assess Google review count and rating against local benchmarks."""
    b_info = state["business_info"]
    name = b_info.get("business_name", "Unknown Business")
    rating = b_info.get("rating", 0.0) or 0.0
    reviews = b_info.get("review_count", 0) or 0
    location = b_info.get("location", "")

    prompt = f"""You are a local SEO consultant evaluating Google Business Profile reputation:
Business: {name}
Location: {location}
Google Rating: {rating} stars
Total Reviews: {reviews} reviews

In 1-2 concise sentences, assess their Google visibility and trust signals. Is the review volume sufficient to win competitive local searches?"""

    assessment = await call_gemini(prompt)
    if not assessment:
        if reviews < 30:
            assessment = f"With only {reviews} reviews at {rating} stars, the business underutilizes social proof and risks losing local Google Map pack rankings to higher-reviewed competitors."
        else:
            assessment = f"Solid foundation of {reviews} reviews at {rating} stars, providing decent credibility."

    return {"review_assessment": assessment.strip()}


async def identify_gaps(state: AgentState) -> Dict[str, Any]:
    """Node 3: Synthesize web and reputation data with RAG context to isolate specific gaps."""
    b_info = state["business_info"]
    a_data = state["analysis_data"]
    rag_context = state.get("rag_context", "")

    name = b_info.get("business_name", "Unknown Business")
    niche = b_info.get("characteristic", "local business")
    rating = b_info.get("rating", 0.0)
    reviews = b_info.get("review_count", 0)

    has_booking = a_data.get("has_online_booking", False)
    has_whatsapp = a_data.get("has_whatsapp", False)
    services = a_data.get("services_found", [])

    prompt = f"""Identify 3-5 specific, high-impact growth gaps for this local business:
Business: {name}
Niche: {niche}
Reviews: {rating} stars with {reviews} reviews
Online Booking: {has_booking}
WhatsApp Chat: {has_whatsapp}
Detected Services: {services[:10]}

Market Context / Regional Competitor Patterns:
{rag_context if rag_context else 'Local competitors in this tier typically leverage 50+ reviews, instant WhatsApp chat, and dedicated high-value service landing pages.'}

Return ONLY a JSON array of strings (no other text). Example format:
["Low review volume (only {reviews} reviews)", "No direct WhatsApp booking button", "Missing service landing pages"]"""

    response = await call_gemini(prompt)
    gaps: List[str] = []

    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            gaps = json.loads(match.group(0))
    except Exception:
        pass

    # Deterministic fallback gaps if LLM failed
    if not gaps:
        if reviews < 30:
            gaps.append(f"Low review volume ({reviews} reviews)")
        if not has_booking:
            gaps.append("No instant online appointment booking")
        if not has_whatsapp:
            gaps.append("No WhatsApp chat option on website")
        if len(services) < 3:
            gaps.append("Missing dedicated landing pages for core high-value services")

    return {"gaps": gaps}


async def diagnose_opportunity(state: AgentState) -> Dict[str, Any]:
    """Node 4: Synthesize gaps into concrete growth opportunities and recommended pitch direction."""
    b_info = state["business_info"]
    gaps = state.get("gaps", [])
    web_assess = state.get("web_presence_assessment", "")
    rev_assess = state.get("review_assessment", "")

    name = b_info.get("business_name", "This business")
    niche = b_info.get("characteristic", "local business")

    prompt = f"""You are a growth marketing agency director.
Diagnose the prime opportunities and recommended pitch for this business:
Business: {name} ({niche})
Identified Gaps: {gaps}
Web Assessment: {web_assess}
Review Assessment: {rev_assess}

Respond with valid JSON with two keys:
1. "opportunities": list of 2-3 strategic opportunity areas (e.g. "Local SEO + Google Business Profile optimization", "Conversion rate optimization with instant messaging")
2. "recommended_actions": a 1-sentence action summary of what to offer them (e.g. "Improve review volume, add WhatsApp booking, and create dedicated high-ticket service pages")

Return ONLY the JSON."""

    response = await call_gemini(prompt)
    opportunities: List[str] = []
    recommended_actions = ""

    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            opportunities = data.get("opportunities", [])
            recommended_actions = data.get("recommended_actions", "")
    except Exception:
        pass

    if not opportunities:
        opportunities = [
            "Local SEO + Google Business Profile optimization",
            "Website lead capture & WhatsApp conversion integration",
        ]
    if not recommended_actions:
        recommended_actions = "Improve review volume, add WhatsApp booking, and enhance core service pages"

    return {
        "opportunities": opportunities,
        "recommended_actions": recommended_actions,
    }


# ---------------------------------------------------------------------------
# Build & Run LangGraph Graph
# ---------------------------------------------------------------------------

def create_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_web_presence", analyze_web_presence)
    graph.add_node("analyze_reviews", analyze_reviews)
    graph.add_node("identify_gaps", identify_gaps)
    graph.add_node("diagnose_opportunity", diagnose_opportunity)

    graph.add_edge(START, "analyze_web_presence")
    graph.add_edge("analyze_web_presence", "analyze_reviews")
    graph.add_edge("analyze_reviews", "identify_gaps")
    graph.add_edge("identify_gaps", "diagnose_opportunity")
    graph.add_edge("diagnose_opportunity", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_agent_graph()
    return _compiled_graph


async def analyze_business(
    business_info: Dict[str, Any],
    analysis_data: Dict[str, Any],
    db: Session,
) -> Dict[str, Any]:
    """Executes the full LangGraph agent workflow for a business."""
    # 1. Retrieve RAG context: query similar market intelligence from pgvector
    niche = business_info.get("characteristic", "local business")
    location = business_info.get("location", "")
    rag_query = f"{niche} in {location} services online booking review count customer acquisition"

    rag_items = await retrieve_similar(query=rag_query, db=db, top_k=4)
    rag_context = "\n---\n".join([item["content_text"] for item in rag_items if item.get("content_text")])

    initial_state: AgentState = {
        "business_info": business_info,
        "analysis_data": analysis_data,
        "rag_context": rag_context,
        "web_presence_assessment": "",
        "review_assessment": "",
        "gaps": [],
        "opportunities": [],
        "recommended_actions": "",
    }

    graph = get_graph()
    final_state = await graph.ainvoke(initial_state)
    return final_state
