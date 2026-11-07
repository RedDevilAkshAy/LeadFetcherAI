from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException

import lead_fetcher.config as _cfg
from lead_fetcher.config import (
    ZOHO_ACCOUNTS_DOMAIN,
    ZOHO_API_DOMAIN,
    ZOHO_CLIENT_ID,
    ZOHO_CLIENT_SECRET,
)
from lead_fetcher.models import BusinessAnalysis, Lead, LeadScore

logger = logging.getLogger("lead_fetcher.crm")

token_cache: Dict[str, Any] = {"access_token": None, "expires_at": 0}

# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

ZOHO_OAUTH_SCOPE = "ZohoCRM.modules.leads.CREATE,ZohoCRM.modules.leads.READ"


def is_zoho_connected() -> bool:
    """Return True if a Zoho refresh token is available at runtime."""
    return bool(_cfg.ZOHO_REFRESH_TOKEN and _cfg.ZOHO_REFRESH_TOKEN.strip())


def get_zoho_auth_url() -> str:
    """Return the Zoho OAuth authorization URL for the user to visit."""
    params = {
        "response_type": "code",
        "client_id": ZOHO_CLIENT_ID,
        "scope": ZOHO_OAUTH_SCOPE,
        "redirect_uri": f"{_cfg.APP_BASE_URL}/zoho/callback",
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{ZOHO_ACCOUNTS_DOMAIN}/oauth/v2/auth?{urlencode(params)}"


async def exchange_code_for_tokens(code: str) -> Dict[str, Any]:
    """Exchange an authorization code for access + refresh tokens."""
    token_url = f"{ZOHO_ACCOUNTS_DOMAIN}/oauth/v2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": _cfg.ZOHO_CLIENT_SECRET,
        "redirect_uri": f"{_cfg.APP_BASE_URL}/zoho/callback",
        "code": code,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            response = await http.post(token_url, data=data)
    except Exception as exc:
        logger.error("Error reaching Zoho Accounts for code exchange: %s", exc)
        raise HTTPException(status_code=502, detail=f"Failed to reach Zoho Accounts: {exc}")

    if response.status_code != 200:
        logger.error("Code exchange failed: HTTP %s — %s", response.status_code, response.text)
        raise HTTPException(
            status_code=502,
            detail=f"Zoho token exchange failed: {response.text}",
        )

    res = response.json()
    if "error" in res:
        raise HTTPException(
            status_code=400,
            detail=f"Zoho OAuth error during code exchange: {res['error']}",
        )

    return res  # contains access_token, refresh_token, expires_in, api_domain, etc.


def save_refresh_token(refresh_token: str) -> None:
    """Persist the Zoho refresh token to the .env file and update the runtime config."""
    # Update runtime config immediately so the server can use it without restart
    _cfg.ZOHO_REFRESH_TOKEN = refresh_token
    # Invalidate the cached access token so the next call uses the new refresh token
    token_cache["access_token"] = None
    token_cache["expires_at"] = 0

    # Persist to .env file
    try:
        env_text = _ENV_PATH.read_text(encoding="utf-8") if _ENV_PATH.exists() else ""
        pattern = re.compile(r"^ZOHO_REFRESH_TOKEN\s*=.*$", re.MULTILINE)
        new_line = f"ZOHO_REFRESH_TOKEN={refresh_token}"
        if pattern.search(env_text):
            env_text = pattern.sub(new_line, env_text)
        else:
            env_text = env_text.rstrip("\n") + f"\n{new_line}\n"
        _ENV_PATH.write_text(env_text, encoding="utf-8")
        logger.info("Zoho refresh token saved to .env")
    except Exception as exc:
        # Non-fatal — token is already in memory, just log a warning
        logger.warning("Could not save refresh token to .env: %s", exc)


async def get_zoho_access_token() -> str:
    """Get a valid Zoho CRM OAuth access token, refreshing using the refresh token when needed."""
    # Read refresh token from runtime config (may be updated after OAuth without restart)
    refresh_token = _cfg.ZOHO_REFRESH_TOKEN
    if not ZOHO_CLIENT_ID or not _cfg.ZOHO_CLIENT_SECRET or not refresh_token:
        raise HTTPException(
            status_code=400,
            detail=(
                "Zoho CRM credentials are not configured. "
                "Please connect Zoho CRM first via the dashboard."
            ),
        )

    # Return cached token if valid for at least 60 more seconds
    if token_cache["access_token"] and time.time() < token_cache["expires_at"] - 60:
        return token_cache["access_token"]

    refresh_url = f"{ZOHO_ACCOUNTS_DOMAIN}/oauth/v2/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": _cfg.ZOHO_CLIENT_SECRET,
        "refresh_token": refresh_token,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            response = await http.post(refresh_url, data=data)
    except Exception as e:
        logger.error(f"Error connecting to Zoho Accounts endpoint: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to reach Zoho Accounts: {e}")

    if response.status_code != 200:
        logger.error(f"Failed to refresh Zoho token: HTTP {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to refresh Zoho token: {response.text}",
        )

    res_data = response.json()
    if "error" in res_data:
        raise HTTPException(
            status_code=400,
            detail=f"Zoho OAuth error: {res_data.get('error')}",
        )

    access_token = res_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=502, detail="Zoho response missing access_token.")

    token_cache["access_token"] = access_token
    token_cache["expires_at"] = time.time() + res_data.get("expires_in", 3600)
    logger.info("Successfully refreshed Zoho CRM access token.")
    return access_token


def _split_owner_name(owner_name: Optional[str], business_name: Optional[str]) -> tuple[str, str]:
    """Split owner name into First_Name and Last_Name (Last_Name is mandatory in Zoho CRM)."""
    if owner_name and owner_name.strip() and owner_name.strip().lower() != "unknown":
        parts = owner_name.strip().split()
        if len(parts) > 1:
            return parts[0], " ".join(parts[1:])
        return "", parts[0]

    # Zoho Leads module requires Last_Name
    fallback_last_name = (business_name or "Business Owner").strip()
    return "", fallback_last_name


def _build_lead_description(
    lead: Lead,
    analysis: Optional[BusinessAnalysis] = None,
    score: Optional[LeadScore] = None,
) -> str:
    """Build a comprehensive diagnostic description to send to Zoho CRM."""
    sections = [
        "=== LEADFETCHER AI INTELLIGENCE REPORT ===",
        f"Business: {lead.business_name or 'N/A'}",
        f"Category: {lead.characteristic or 'Local Business'}",
        f"Location: {lead.location or 'N/A'}",
        f"Website: {lead.business_website or 'N/A'}",
    ]

    if lead.business_profile:
        sections.append(f"Google Places Profile: {lead.business_profile}")

    if lead.rating:
        sections.append(f"Rating: {lead.rating} stars ({lead.review_count or 0} reviews)")

    if lead.website_email:
        sections.append(f"Crawled Website Email: {lead.website_email}")

    if analysis and analysis.website_summary:
        sections.append(f"\nWebsite Summary:\n{analysis.website_summary}")

    if score:
        sections.append(f"\nOpportunity Score: {score.score or 0}/100 | Priority: {score.priority or 'N/A'}")
        if score.opportunities:
            try:
                opps = json.loads(score.opportunities)
                sections.append(f"Growth Gaps: {', '.join(opps)}")
            except Exception:
                sections.append(f"Growth Gaps: {score.opportunities}")

        if score.recommended_actions:
            sections.append(f"Recommended Strategy: {score.recommended_actions}")

        if score.pitch_whatsapp:
            sections.append(f"\n--- WhatsApp Pitch ---\n{score.pitch_whatsapp}")

        if score.pitch_email:
            sections.append(f"\n--- Email Pitch ---\n{score.pitch_email}")

    return "\n".join(sections)


async def push_lead_to_zoho(
    lead: Lead,
    analysis: Optional[BusinessAnalysis] = None,
    score: Optional[LeadScore] = None,
) -> Dict[str, Any]:
    """Create a new Lead in Zoho CRM Leads module."""
    # Email is strictly required
    email = (lead.email or lead.website_email or "").strip()
    if not email:
        raise HTTPException(
            status_code=400,
            detail="Cannot send to CRM: No email address available for this business.",
        )

    first_name, last_name = _split_owner_name(lead.owner_name, lead.business_name)
    description = _build_lead_description(lead, analysis, score)

    zoho_lead: Dict[str, Any] = {
        "Company": lead.business_name or "Unknown Company",
        "Last_Name": last_name,
        "First_Name": first_name if first_name else None,
        "Email": email,
        "Phone": lead.number,
        "Website": lead.business_website,
        "City": lead.location,
        "Lead_Source": "LeadFetcher AI",
        "Description": description,
    }

    # Remove empty/None fields
    zoho_lead = {k: v for k, v in zoho_lead.items() if v not in (None, "")}

    access_token = await get_zoho_access_token()
    url = f"{ZOHO_API_DOMAIN}/crm/v8/Leads"
    payload = {"data": [zoho_lead]}
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            response = await http.post(url, headers=headers, json=payload)
    except Exception as e:
        logger.error(f"Error connecting to Zoho CRM API: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to reach Zoho CRM API: {e}")

    if response.status_code not in (200, 201):
        logger.error(f"Zoho Lead creation failed: HTTP {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=response.status_code if response.status_code < 500 else 502,
            detail=f"Zoho CRM error: {response.text}",
        )

    response_data = response.json()
    data_list = response_data.get("data", [])
    if data_list:
        record_res = data_list[0]
        if record_res.get("code") == "SUCCESS":
            zoho_id = record_res.get("details", {}).get("id")
            logger.info(f"Successfully pushed Lead {lead.id} to Zoho CRM [Zoho ID: {zoho_id}]")
            return {
                "success": True,
                "message": f"Lead successfully created in Zoho CRM (ID: {zoho_id})",
                "zoho_lead_id": zoho_id,
                "zoho_response": record_res,
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Zoho CRM rejected record: {record_res.get('message', 'Unknown error')} ({record_res.get('code')})",
            )

    return {
        "success": True,
        "message": "Lead request sent to Zoho CRM",
        "zoho_response": response_data,
    }
