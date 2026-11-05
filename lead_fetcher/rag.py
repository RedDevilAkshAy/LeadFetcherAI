from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from openai import OpenAI

try:
    from lead_fetcher.config import GEMINI_API_KEY, GEMINI_MODEL, NVIDIA_API_KEY, NVIDIA_EMBEDDING_MODEL, NVIDIA_MODEL, NVIDIA_BASE_URL
    from lead_fetcher.models import BusinessEmbedding
except ImportError:
    from config import GEMINI_API_KEY, GEMINI_MODEL, NVIDIA_API_KEY, NVIDIA_EMBEDDING_MODEL, NVIDIA_MODEL, NVIDIA_BASE_URL
    from models import BusinessEmbedding

logger = logging.getLogger("lead_fetcher.rag")

_agent_nim_client = None

def get_agent_client():
    global _agent_nim_client

    if _agent_nim_client is None and NVIDIA_API_KEY:
        try:
            _agent_nim_client = OpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=NVIDIA_API_KEY,
            )
        except Exception as e:
            logger.warning(f"Could not initialize NVIDIA NIM client for RAG: {e}")


    return _agent_nim_client


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits plain text into overlapping chunks."""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# async def embed_text(text: str) -> Optional[List[float]]:
#     """Generates an embedding vector for given text using Gemini Embeddings API."""
#     client = get_agent_client()
#     if not client or not text.strip():
#         return None

#     model = NVIDIA_EMBEDDING_MODEL

#     try:
#         # Use native async if supported by google-genai
#         if hasattr(client, "aio") and hasattr(client.aio, "models"):
#             response = await client.aio.models.embed_content(
#                 model=model,
#                 contents=text,
#             )
#         else:
#             response = await asyncio.to_thread(
#                 client.models.embed_content,
#                 model=model,
#                 contents=text,
#             )

#         if hasattr(response, "embeddings") and response.embeddings:
#             return response.embeddings[0].values
#         elif hasattr(response, "embedding") and response.embedding:
#             return response.embedding.values
#         return None

#     except Exception as e:
#         logger.warning(f"Failed to generate embedding for text ({len(text)} chars): {e}")
#         return None

async def embed_text(
    text: str,
    input_type: Literal["passage", "query"] = "passage",
) -> Optional[List[float]]:
    """Generates an embedding vector using the NVIDIA NIM embeddings API."""
    if not text or not text.strip():
        return None

    if input_type not in ("passage", "query"):
        raise ValueError("input_type must be 'passage' or 'query'")

    if not NVIDIA_EMBEDDING_MODEL:
        logger.error("NVIDIA_EMBEDDING_MODEL is not configured")
        return None

    client = get_agent_client()
    if not client:
        return None

    try:
        response = await asyncio.to_thread(
            client.embeddings.create,
            model=NVIDIA_EMBEDDING_MODEL,
            input=[text.strip()],
            encoding_format="float",
            extra_body={
                "input_type": input_type,
                "truncate": "END",
            },
        )

        if not response.data:
            logger.warning("NVIDIA NIM returned no embedding data")
            return None

        return list(response.data[0].embedding)

    except Exception as e:
        logger.warning(
            "Failed to generate NVIDIA embedding for text (%d chars, input_type=%s): %s",
            len(text),
            input_type,
            e,
        )
        return None

async def store_embeddings(lead_id: int, content_chunks: List[dict], db: Session) -> None:
    """Stores multiple chunks and their pgvector embeddings for a specific lead."""
    if not content_chunks:
        return

    try:
        # Clear previous embeddings for this lead
        db.execute(delete(BusinessEmbedding).where(BusinessEmbedding.lead_id == lead_id))
        db.commit()

        for item in content_chunks:
            c_type = item.get("content_type", "general")
            c_text = item.get("text", "").strip()
            if not c_text:
                continue

            vector = await embed_text(c_text)
            embedding_record = BusinessEmbedding(
                lead_id=lead_id,
                content_type=c_type,
                content_text=c_text,
                embedding=vector,
            )
            db.add(embedding_record)

        db.commit()
        logger.info(f"Stored {len(content_chunks)} embedding chunks for lead {lead_id}")

    except Exception as e:
        db.rollback()
        logger.error(f"Error storing embeddings for lead {lead_id}: {e}")


async def retrieve_similar(
    query: str,
    db: Session,
    lead_id: Optional[int] = None,
    top_k: int = 5,
) -> List[dict]:
    """Cosine similarity search using pgvector.

    Can search across all businesses (to find regional market norms / competitors)
    or scoped to a single lead.
    """
    query_vector = await embed_text(query)

    results: List[dict] = []
    try:
        # Check if database supports vector operations
        stmt = select(BusinessEmbedding)
        if lead_id is not None:
            stmt = stmt.where(BusinessEmbedding.lead_id == lead_id)

        if query_vector is not None and hasattr(BusinessEmbedding.embedding, "cosine_distance"):
            try:
                # pgvector cosine distance
                stmt = stmt.order_by(BusinessEmbedding.embedding.cosine_distance(query_vector)).limit(top_k)
                rows = db.scalars(stmt).all()
                for row in rows:
                    results.append({
                        "lead_id": row.lead_id,
                        "content_type": row.content_type,
                        "content_text": row.content_text,
                    })
                return results
            except Exception as vector_err:
                logger.warning(f"pgvector query failed, falling back to text query: {vector_err}")

        # Fallback query when vector extension is not available or query_vector is None
        stmt = stmt.limit(top_k)
        rows = db.scalars(stmt).all()
        for row in rows:
            results.append({
                "lead_id": row.lead_id,
                "content_type": row.content_type,
                "content_text": row.content_text,
            })

    except Exception as e:
        logger.error(f"Error in retrieve_similar: {e}")

    return results
