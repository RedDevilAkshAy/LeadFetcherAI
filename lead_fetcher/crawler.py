"""
Asynchronous website crawler for local business lead intelligence extraction.

Crawls a business website (any niche) to extract structured signals used for
lead scoring: services offered, online-booking presence, WhatsApp availability,
contact forms, pricing information, and social-media links.

Usage::

    import asyncio
    from lead_fetcher.crawler import crawl_website

    result = asyncio.run(crawl_website("https://example-dental.com"))
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_REQUEST_TIMEOUT = 5.0  # seconds per page
_INTER_PAGE_DELAY = 1.0  # polite crawl delay (robots.txt etiquette)
_MAX_INTERNAL_PAGES = 5  # max additional pages beyond the landing page

# Navigation link keywords used to discover high-value internal pages.
_NAV_KEYWORDS: set[str] = {
    "about", "service", "services", "pricing", "price", "prices", "rates",
    "contact", "book", "booking", "appointment", "schedule", "menu",
    "treatments", "classes", "programs", "packages", "plans", "team",
    "our-team", "staff", "faq", "gallery", "testimonials", "reviews",
    "offers", "specials", "hours",
}

# Booking-related keywords (case-insensitive).
_BOOKING_KEYWORDS: re.Pattern[str] = re.compile(
    r"book\s*(?:now|online|an?\s*appointment)|schedule\s*(?:now|online|an?\s*appointment)"
    r"|request\s*an?\s*appointment|make\s*an?\s*appointment|reserve\s*(?:now|online)"
    r"|online\s*booking|book\s*a\s*(?:session|class|table|visit|consultation)",
    re.IGNORECASE,
)

# Known booking-platform domains found in iframes / script src / links.
_BOOKING_PLATFORMS: re.Pattern[str] = re.compile(
    r"calendly\.com|acuityscheduling\.com|zocdoc\.com|booksy\.com"
    r"|vagaro\.com|mindbodyonline\.com|mindbody\.io|squareup\.com/appointments"
    r"|setmore\.com|appointy\.com|fresha\.com|treatwell\.com"
    r"|simplybook\.me|genbook\.com|schedulicity\.com|styleseat\.com"
    r"|hotdoc\.com\.au|cliniko\.com|jane\.app|resy\.com|opentable\.com"
    r"|yelp\.com/biz/.*/reservations",
    re.IGNORECASE,
)

# WhatsApp detection patterns.
_WHATSAPP_PATTERN: re.Pattern[str] = re.compile(
    r"wa\.me/|api\.whatsapp\.com|whatsapp\.com/send|whatsapp",
    re.IGNORECASE,
)

# Social-media platform detection (domain → canonical name).
_SOCIAL_DOMAINS: dict[str, str] = {
    "facebook.com": "facebook",
    "fb.com": "facebook",
    "instagram.com": "instagram",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "linkedin.com": "linkedin",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "tiktok.com": "tiktok",
    "pinterest.com": "pinterest",
    "yelp.com": "yelp",
}

# Pricing-related keywords.
_PRICING_KEYWORDS: re.Pattern[str] = re.compile(
    r"\$\s*\d+|\d+\s*(?:USD|EUR|GBP|AUD|CAD|INR|AED)"
    r"|pricing|price\s*list|rate\s*card|our\s*(?:rates|fees|prices)"
    r"|cost\s*(?:of|for)|starts?\s*(?:at|from)\s*\$"
    r"|per\s*(?:session|hour|visit|class|month)",
    re.IGNORECASE,
)

# Tags / elements that are considered "noise" for text extraction.
_STRIP_TAGS: set[str] = {
    "script", "style", "noscript", "svg", "img", "video", "audio",
    "iframe", "canvas", "map", "object", "embed",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CrawlResult:
    """Structured intelligence extracted from a business website."""

    services_found: list[str] = field(default_factory=list)
    has_online_booking: bool = False
    has_whatsapp: bool = False
    has_contact_form: bool = False
    pricing_found: bool = False
    social_links: dict[str, str] = field(default_factory=dict)
    crawl_text: str = ""
    website_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_url(base: str, href: str) -> str | None:
    """Resolve *href* against *base* and return a clean absolute URL.

    Returns ``None`` when the resulting URL is off-site, is a fragment-only
    link, or uses a non-HTTP(S) scheme.
    """
    if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
        return None
    absolute = urljoin(base, href)
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None
    # Strip fragment
    return parsed._replace(fragment="").geturl()


def _same_domain(url_a: str, url_b: str) -> bool:
    """Check whether two URLs share the same registered domain."""
    host_a = urlparse(url_a).hostname or ""
    host_b = urlparse(url_b).hostname or ""
    # Simple suffix comparison (handles www. prefix differences).
    return host_a.replace("www.", "") == host_b.replace("www.", "")


def _extract_clean_text(soup: BeautifulSoup) -> str:
    """Return visible text from *soup*, stripping noise elements."""
    # Work on a copy so we don't mutate the tree for further analysis.
    clone = BeautifulSoup(str(soup), "lxml")
    for tag_name in _STRIP_TAGS:
        for el in clone.find_all(tag_name):
            el.decompose()
    text = clone.get_text(separator=" ", strip=True)
    # Collapse whitespace runs.
    return re.sub(r"\s{2,}", " ", text).strip()


def _discover_nav_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Find internal page URLs worth crawling from <nav>, <header>, and <a> tags."""
    candidates: dict[str, str] = {}  # url -> anchor-text (dedup)

    # Prioritise links inside <nav> and <header>; fall back to all <a>.
    containers: list[Tag] = soup.find_all(["nav", "header"])  # type: ignore[assignment]
    anchors: list[Tag] = []
    for container in containers:
        anchors.extend(container.find_all("a", href=True))
    if not anchors:
        anchors = soup.find_all("a", href=True)  # type: ignore[assignment]

    for a_tag in anchors:
        href = a_tag.get("href", "")
        if isinstance(href, list):
            href = href[0]
        norm = _normalise_url(base_url, href)
        if norm is None or not _same_domain(base_url, norm):
            continue
        anchor_text = a_tag.get_text(strip=True).lower()
        path = urlparse(norm).path.strip("/").lower().replace("-", "").replace("_", "")
        # Check if the anchor text or URL path matches any of our nav keywords.
        matched = any(
            kw in anchor_text or kw.replace("-", "") in path
            for kw in _NAV_KEYWORDS
        )
        if matched and norm not in candidates:
            candidates[norm] = anchor_text

    # Deduplicate & limit.
    return list(candidates.keys())[:_MAX_INTERNAL_PAGES]


def _detect_services(soup: BeautifulSoup, text: str) -> list[str]:
    """Heuristically extract a list of service/offering names.

    Strategy:
    1. Look for lists (<ul>/<ol>) inside sections whose heading contains
       keywords like "services", "treatments", "what we offer", etc.
    2. Collect items from those lists.
    3. Fall back to heading tags (h2/h3) that look service-like.
    """
    services: list[str] = []
    seen: set[str] = set()

    service_heading_re = re.compile(
        r"service|treatment|what we (?:do|offer)|our (?:services|offerings|specialties|programs|classes|menu)"
        r"|we (?:offer|provide|specialize)",
        re.IGNORECASE,
    )

    # --- Strategy 1: lists under service-related headings ---------------
    for heading in soup.find_all(re.compile(r"^h[1-4]$")):
        heading_text = heading.get_text(strip=True)
        if not service_heading_re.search(heading_text):
            continue
        # Walk siblings until next heading to find <ul>/<ol>.
        sibling: Tag | None = heading.find_next_sibling()
        while sibling and sibling.name not in {"h1", "h2", "h3", "h4"}:
            if sibling.name in ("ul", "ol"):
                for li in sibling.find_all("li"):
                    item = li.get_text(strip=True)
                    if 3 < len(item) < 120 and item.lower() not in seen:
                        seen.add(item.lower())
                        services.append(item)
            sibling = sibling.find_next_sibling()  # type: ignore[assignment]

    # --- Strategy 2: headings themselves --------------------------------
    if not services:
        for heading in soup.find_all(re.compile(r"^h[2-4]$")):
            h_text = heading.get_text(strip=True)
            if 3 < len(h_text) < 80:
                lower = h_text.lower()
                # Skip generic headings.
                if lower in seen or any(
                    skip in lower
                    for skip in (
                        "about", "contact", "home", "welcome", "blog",
                        "testimonial", "review", "faq", "location", "hour",
                        "follow", "connect", "copyright", "privacy", "terms",
                    )
                ):
                    continue
                seen.add(lower)
                services.append(h_text)

    # Cap the list to avoid noise.
    return services[:25]


def _detect_online_booking(soup: BeautifulSoup, text: str) -> bool:
    """Return ``True`` if the page signals an online booking capability."""
    # Keyword search in visible text.
    if _BOOKING_KEYWORDS.search(text):
        return True
    # Check for booking-platform embeds (iframes, scripts, links).
    for tag in soup.find_all(["iframe", "script", "a", "link"]):
        src = tag.get("src") or tag.get("href") or ""
        if isinstance(src, list):
            src = " ".join(src)
        if _BOOKING_PLATFORMS.search(src):
            return True
    # Check data attributes & class names.
    for tag in soup.find_all(True):
        attrs_blob = " ".join(
            str(v) for v in tag.attrs.values() if isinstance(v, (str, list))
        )
        if _BOOKING_PLATFORMS.search(attrs_blob):
            return True
    return False


def _detect_whatsapp(soup: BeautifulSoup, text: str) -> bool:
    """Return ``True`` if a WhatsApp link or widget is detected."""
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        if isinstance(href, list):
            href = href[0]
        if re.search(r"wa\.me/|api\.whatsapp\.com|whatsapp\.com/send", href, re.IGNORECASE):
            return True
    # Check iframes / scripts for WhatsApp widgets.
    for tag in soup.find_all(["iframe", "script", "div"]):
        src = tag.get("src") or tag.get("data-src") or ""
        class_list = " ".join(tag.get("class", []))
        id_attr = tag.get("id", "") or ""
        blob = f"{src} {class_list} {id_attr}"
        if _WHATSAPP_PATTERN.search(blob):
            return True
    return False


def _detect_contact_form(soup: BeautifulSoup) -> bool:
    """Return ``True`` if a contact/enquiry form is detected."""
    forms = soup.find_all("form")
    for form in forms:
        inputs_text = " ".join(
            (inp.get("name", "") or "") + " " + (inp.get("placeholder", "") or "") + " " + (inp.get("type", "") or "")
            for inp in form.find_all(["input", "textarea", "select"])
        ).lower()
        # Heuristic: a contact form usually has an email or message field.
        if re.search(r"email|message|enquiry|inquiry|comment|question|subject|your.?name", inputs_text):
            return True
    return False


def _detect_pricing(soup: BeautifulSoup, text: str) -> bool:
    """Return ``True`` if pricing information is detected."""
    if _PRICING_KEYWORDS.search(text):
        return True
    # Check for pricing tables.
    for table in soup.find_all("table"):
        table_text = table.get_text(strip=True).lower()
        if re.search(r"price|cost|rate|fee|\$\d", table_text):
            return True
    return False


def _extract_social_links(soup: BeautifulSoup) -> dict[str, str]:
    """Return a mapping of social platform names to profile URLs."""
    socials: dict[str, str] = {}
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        if isinstance(href, list):
            href = href[0]
        if not href or not href.startswith("http"):
            continue
        hostname = (urlparse(href).hostname or "").replace("www.", "").lower()
        for domain, platform in _SOCIAL_DOMAINS.items():
            if hostname.endswith(domain) and platform not in socials:
                socials[platform] = href
                break
    return socials


def _build_summary(result: CrawlResult) -> str:
    """Generate a concise one-line website summary."""
    parts: list[str] = []

    if result.services_found:
        parts.append(f"{len(result.services_found)} services detected")
    else:
        parts.append("No clear services listing")

    positives: list[str] = []
    negatives: list[str] = []

    for label, flag in [
        ("online booking", result.has_online_booking),
        ("WhatsApp", result.has_whatsapp),
        ("contact form", result.has_contact_form),
        ("pricing info", result.pricing_found),
    ]:
        (positives if flag else negatives).append(label)

    if positives:
        parts.append("has " + ", ".join(positives))
    if negatives:
        parts.append("missing " + ", ".join(negatives))

    if result.social_links:
        parts.append(f"{len(result.social_links)} social links")

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Core fetch helper
# ---------------------------------------------------------------------------

async def _fetch_page(
    client: httpx.AsyncClient,
    url: str,
) -> BeautifulSoup | None:
    """Fetch a single page and return parsed soup, or ``None`` on failure."""
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            logger.debug("Skipping non-HTML content at %s (type=%s)", url, content_type)
            return None
        return BeautifulSoup(response.text, "lxml")
    except httpx.TimeoutException:
        logger.warning("Timeout fetching %s", url)
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %s for %s", exc.response.status_code, url)
    except httpx.RequestError as exc:
        logger.warning("Request error for %s: %s", url, exc)
    except Exception:
        logger.exception("Unexpected error fetching %s", url)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def crawl_website(url: str) -> dict[str, Any]:
    """Crawl a local-business website and extract lead-scoring intelligence.

    Fetches the landing page plus up to 5 high-value internal pages
    (services, about, contact, pricing, etc.) discovered from navigation
    links.  Extracts structured signals and combined page text suitable for
    RAG embedding.

    Args:
        url: The root URL of the business website (e.g. ``"https://example.com"``).

    Returns:
        A dictionary containing the fields defined by :class:`CrawlResult`.
    """
    logger.info("Starting crawl for %s", url)

    # Normalise the root URL.
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"

    result = CrawlResult()
    all_text_parts: list[str] = []
    visited: set[str] = set()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(_REQUEST_TIMEOUT),
        headers={"User-Agent": _USER_AGENT},
        limits=httpx.Limits(max_connections=5),
    ) as client:
        # ------ Fetch landing page --------------------------------------
        soup = await _fetch_page(client, url)
        if soup is None:
            logger.error("Failed to fetch landing page %s — aborting crawl", url)
            result.website_summary = "Website unreachable or returned non-HTML content."
            return result.to_dict()

        visited.add(url)
        page_text = _extract_clean_text(soup)
        all_text_parts.append(page_text)

        # Analyse landing page.
        result.services_found.extend(_detect_services(soup, page_text))
        result.has_online_booking = _detect_online_booking(soup, page_text)
        result.has_whatsapp = _detect_whatsapp(soup, page_text)
        result.has_contact_form = _detect_contact_form(soup)
        result.pricing_found = _detect_pricing(soup, page_text)
        result.social_links.update(_extract_social_links(soup))

        # ------ Discover & fetch internal pages -------------------------
        internal_urls = _discover_nav_links(soup, url)
        logger.info("Discovered %d internal pages to crawl", len(internal_urls))

        for page_url in internal_urls:
            if page_url in visited:
                continue
            visited.add(page_url)

            # Polite delay between requests.
            await asyncio.sleep(_INTER_PAGE_DELAY)

            page_soup = await _fetch_page(client, page_url)
            if page_soup is None:
                continue

            page_text = _extract_clean_text(page_soup)
            all_text_parts.append(page_text)

            # Accumulate detections (only flip flags to True, never back).
            new_services = _detect_services(page_soup, page_text)
            for svc in new_services:
                if svc.lower() not in {s.lower() for s in result.services_found}:
                    result.services_found.append(svc)

            if not result.has_online_booking:
                result.has_online_booking = _detect_online_booking(page_soup, page_text)
            if not result.has_whatsapp:
                result.has_whatsapp = _detect_whatsapp(page_soup, page_text)
            if not result.has_contact_form:
                result.has_contact_form = _detect_contact_form(page_soup)
            if not result.pricing_found:
                result.pricing_found = _detect_pricing(page_soup, page_text)

            # Merge socials (first-seen wins per platform).
            for platform, link in _extract_social_links(page_soup).items():
                result.social_links.setdefault(platform, link)

    # Cap services list.
    result.services_found = result.services_found[:25]

    # Combine text from all pages.
    result.crawl_text = "\n\n".join(all_text_parts)

    # Generate summary.
    result.website_summary = _build_summary(result)

    logger.info(
        "Crawl complete for %s — %d pages, %d services, booking=%s, whatsapp=%s",
        url,
        len(visited),
        len(result.services_found),
        result.has_online_booking,
        result.has_whatsapp,
    )

    return result.to_dict()
