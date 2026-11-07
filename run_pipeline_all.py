import asyncio
import logging

from lead_fetcher.pipeline import process_lead_pipeline
from database import SessionLocal
from lead_fetcher.models import Lead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("run_pipeline_all")

MAX_CONCURRENT_LEADS = 5


async def process_one_lead(lead_id: int, index: int, total: int, semaphore: asyncio.Semaphore):
    async with semaphore:
        db = SessionLocal()

        try:
            logger.info(
                "Processing lead %d/%d: id=%s",
                index,
                total,
                lead_id,
            )

            result = await process_lead_pipeline(
                lead_id=lead_id,
                db=db,
            )

            if result is None:
                logger.warning("No result returned for lead id=%s", lead_id)
            else:
                logger.info("Completed lead id=%s", lead_id)

            return result

        except Exception:
            logger.exception("Pipeline failed for lead id=%s", lead_id)
            db.rollback()
            return None

        finally:
            db.close()


async def run_all_leads():
    db = SessionLocal()

    try:
        leads = (
            db.query(Lead)
            .filter(~Lead.embeddings.any())
            .all()
        )
        lead_ids = [lead.id for lead in leads]
        total = len(lead_ids)

        logger.info("Found %d leads", total)

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LEADS)

        tasks = [
            process_one_lead(
                lead_id=lead_id,
                index=index,
                total=total,
                semaphore=semaphore,
            )
            for index, lead_id in enumerate(lead_ids, start=1)
        ]

        results = await asyncio.gather(*tasks)

        completed = sum(result is not None for result in results)
        failed = total - completed

        logger.info(
            "Finished: %d completed, %d failed, %d total",
            completed,
            failed,
            total,
        )

        return results

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(run_all_leads())