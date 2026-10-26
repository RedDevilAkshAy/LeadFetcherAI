from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Mock third-party libraries not present in global Python 3.8 environment
for mod in [
    "httpx",
    "google",
    "google.genai",
    "google.genai.types",
    "sqlalchemy",
    "sqlalchemy.orm",
    "sqlalchemy.exc",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import unittest
from lead_fetcher.sourcer import LeadFinder


class TestLeadFinder(unittest.TestCase):
    def setUp(self):
        self.finder = LeadFinder(
            niche="Aesthetic clinics",
            locations=["Dubai, UAE"],
            google_api_key="fake-google-key",
            gemini_api_key="fake-gemini-key",
        )

    def test_normalize_domain(self):
        self.assertEqual(
            self.finder.normalize_domain("https://www.example.com/path"), "example.com"
        )
        self.assertEqual(
            self.finder.normalize_domain("http://clinic.ae/"), "clinic.ae"
        )
        self.assertEqual(
            self.finder.normalize_domain("sub.domain.com:8080/page"), "sub.domain.com"
        )
        self.assertEqual(
            self.finder.normalize_domain("www.dermacare.com"), "dermacare.com"
        )
        self.assertIsNone(self.finder.normalize_domain(""))
        self.assertIsNone(self.finder.normalize_domain(None))

    def test_sanitize_owner_name(self):
        self.assertEqual(self.finder.sanitize_owner_name("John Doe"), "John Doe")
        self.assertEqual(
            self.finder.sanitize_owner_name("Owner: Jane Smith"), "Jane Smith"
        )
        self.assertEqual(
            self.finder.sanitize_owner_name("Founder: Dr. Alex Turner."),
            "Dr. Alex Turner",
        )
        self.assertEqual(
            self.finder.sanitize_owner_name("**Michael Scott**"), "Michael Scott"
        )
        self.assertEqual(self.finder.sanitize_owner_name("Unknown."), "Unknown")
        self.assertEqual(
            self.finder.sanitize_owner_name("The owner is unknown"), "Unknown"
        )
        self.assertEqual(self.finder.sanitize_owner_name(""), "Unknown")

        # Column max length 50 chars
        long_name = "A" * 80
        sanitized = self.finder.sanitize_owner_name(long_name)
        self.assertLessEqual(len(sanitized), 50)

    def test_save_lead_to_db(self):
        mock_db = MagicMock()
        mock_lead_cls = MagicMock()

        with patch("lead_fetcher.sourcer.Lead", mock_lead_cls):
            business = {
                "name": "Super Clinic",
                "url": "https://superclinic.ae",
                "phone": "+971 4 000 0000",
                "business_type": "Medical clinic",
            }
            lead = self.finder.save_lead_to_db(
                business=business,
                owner_name="Dr. Ahmed",
                niche="Aesthetic clinics",
                db=mock_db,
            )
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once()

    def test_save_lead_truncates_fields_safely(self):
        mock_db = MagicMock()
        mock_lead_cls = MagicMock()

        with patch("lead_fetcher.sourcer.Lead", mock_lead_cls) as mock_cls:
            business = {
                "name": "N" * 150,
                "url": "https://" + ("U" * 150) + ".com",
                "phone": "+971" + ("9" * 30),
                "business_type": "T" * 50,
            }
            self.finder.save_lead_to_db(
                business=business,
                owner_name="O" * 80,
                niche="Niche Name Very Long",
                db=mock_db,
            )
            mock_cls.assert_called_once()
            _, kwargs = mock_cls.call_args
            self.assertLessEqual(len(kwargs["business_name"]), 100)
            self.assertLessEqual(len(kwargs["business_website"]), 100)
            self.assertLessEqual(len(kwargs["owner_name"]), 50)
            self.assertLessEqual(len(kwargs["number"]), 20)
            self.assertLessEqual(len(kwargs["characteristic"]), 20)
            self.assertIsNone(kwargs["email"])


if __name__ == "__main__":
    unittest.main()
