"""Script to clean existing owner_name sentences in the database into single full names

WITHOUT calling the Gemini API.
"""

from __future__ import annotations

import logging
import sys

from database import SessionLocal
from lead_fetcher.models import Lead
from lead_fetcher.sourcer import LeadFinder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("clean_db_leads")


def clean_database_leads():
    finder = LeadFinder()
    db = SessionLocal()

    try:
        leads = db.query(Lead).all()
        logger.info(f"Loaded {len(leads)} leads from the database.")

        updated_count = 0
        for lead in leads:
            old_name = lead.owner_name
            if not old_name or old_name == "Unknown":
                continue

            cleaned = finder.clean_owner_name(old_name, lead.business_name or "")

            if cleaned != old_name:
                lead.owner_name = cleaned
                updated_count += 1
                logger.info(
                    f"Updated [ID {lead.id}] '{lead.business_name}':\n"
                    f"  BEFORE: '{old_name}'\n"
                    f"  AFTER:  '{cleaned}'\n"
                )

        if updated_count > 0:
            db.commit()
            logger.info(
                f"\nSUCCESS: Updated {updated_count} lead(s) in the database without calling Gemini API."
            )
        else:
            logger.info("All leads already have clean owner names. No changes needed.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during database cleanup: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    clean_database_leads()
