from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from lead_fetcher.crawler import (
    CrawlResult,
    _detect_whatsapp,
    _detect_online_booking,
    _detect_pricing,
    _detect_services,
    _extract_social_links,
)
from lead_fetcher.rag import chunk_text
from lead_fetcher.scorer import score_lead
from lead_fetcher.pitch import _get_salutation, _fallback_pitches
from server import format_lead_card
from lead_fetcher.models import Lead, BusinessAnalysis, LeadScore


class TestPipelineComponents(unittest.TestCase):
    def test_chunk_text(self):
        text = "word " * 1200
        chunks = chunk_text(text, chunk_size=300, overlap=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.split()) <= 300 for c in chunks))

    def test_crawler_html_parsing(self):
        sample_html = """
        <html>
            <head><title>SmileCare Dental</title></head>
            <body>
                <a href="https://wa.me/919876543210">Chat on WhatsApp</a>
                <a href="https://instagram.com/smilecaredental">Instagram</a>
                <h2>Our Services</h2>
                <ul>
                    <li>Teeth Whitening</li>
                    <li>Dental Implants</li>
                    <li>Root Canal Therapy</li>
                </ul>
                <div class="pricing-card">Root Canal: $300</div>
            </body>
        </html>
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sample_html, "lxml")
        text = soup.get_text()

        self.assertTrue(_detect_whatsapp(soup, text))
        self.assertFalse(_detect_online_booking(soup, text))
        self.assertTrue(_detect_pricing(soup, text))

        socials = _extract_social_links(soup)
        self.assertEqual(socials.get("instagram"), "https://instagram.com/smilecaredental")

    def test_score_lead_high_priority(self):
        biz_info = {
            "business_name": "SmileCare Dental",
            "rating": 4.2,
            "review_count": 18,
            "business_website": "https://smilecaredental.in"
        }
        analysis_data = {
            "has_online_booking": False,
            "has_whatsapp": False,
            "has_contact_form": False,
            "pricing_found": False,
            "services_found": ["Cleaning"],
            "social_links": {"instagram": None, "facebook": None}
        }
        agent_result = {
            "gaps": ["Low reviews", "No WhatsApp booking", "No online calendar"],
            "opportunities": ["Local SEO", "WhatsApp booking"],
            "recommended_actions": "Improve review volume and add WhatsApp booking"
        }

        result = score_lead(biz_info, analysis_data, agent_result)
        self.assertGreaterEqual(result["score"], 65.0)
        self.assertEqual(result["priority"], "High")

    def test_pitch_salutation_and_fallback(self):
        # Dental clinic with doctor name
        sal = _get_salutation("Rajesh Kumar", "dental clinics")
        self.assertEqual(sal, "Hi Dr. Rajesh Kumar,")

        # Dental clinic with Unknown name
        sal_unk = _get_salutation("Unknown", "dental clinics")
        self.assertEqual(sal_unk, "Hi Doctor,")

        # General business
        sal_gen = _get_salutation("Sarah Jenkins", "hair salons")
        self.assertEqual(sal_gen, "Hi Sarah Jenkins,")

        # Fallback pitch test
        biz_info = {
            "business_name": "SmileCare Dental",
            "owner_name": "Dr. Rajesh",
            "characteristic": "dental clinic",
            "review_count": 18,
            "rating": 4.2,
        }
        analysis_data = {
            "has_whatsapp": False,
            "has_online_booking": False,
        }
        agent_result = {"gaps": ["only 18 reviews"]}

        pitches = _fallback_pitches(biz_info, analysis_data, agent_result)
        self.assertIn("SmileCare Dental", pitches["whatsapp"])
        self.assertIn("18 Google reviews", pitches["whatsapp"])
        self.assertIn("free 10-minute audit", pitches["whatsapp"])
        self.assertIn("Subject:", pitches["email"])

    def test_format_lead_card(self):
        lead = Lead(
            business_name="SmileCare Dental",
            location="Kakkanad",
            rating=4.2,
            review_count=18,
        )
        analysis = BusinessAnalysis(
            website_summary="Present, but no online booking or WhatsApp button",
        )
        score = LeadScore(
            priority="High",
            opportunities='["Local SEO", "Google Business Profile optimization"]',
            recommended_actions="Improve review volume, add WhatsApp booking, and create service pages for implants/aligners",
            pitch_whatsapp="Hi Dr. Rajesh, I came across SmileCare Dental on Google...",
        )

        card = format_lead_card(lead, analysis, score)
        self.assertIn("Business: SmileCare Dental, Kakkanad", card)
        self.assertIn("Website: Present, but no online booking or WhatsApp button", card)
        self.assertIn("Reviews: 4.2 stars, 18 reviews", card)
        self.assertIn("Priority: High", card)
        self.assertIn("Generated message:", card)


if __name__ == "__main__":
    unittest.main()
