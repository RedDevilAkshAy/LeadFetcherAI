import os
import logging
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

logger = logging.getLogger("database")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./leads.db"

engine = create_engine(DATABASE_URL, echo=False)

# Enable pgvector extension when connecting to PostgreSQL
if "postgresql" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def enable_pgvector(dbapi_conn, connection_record):
        try:
            with dbapi_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            dbapi_conn.commit()
        except Exception as e:
            logger.warning(f"Failed to auto-enable vector extension: {e}")

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables if they do not exist."""
    import lead_fetcher.models  # ensure models are registered
    Base.metadata.create_all(bind=engine)
