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

_vector_available = None


def is_vector_available() -> bool:
    """Checks whether the 'vector' extension is installed and available in the database."""
    global _vector_available
    if _vector_available is not None:
        return _vector_available

    if "postgresql" not in DATABASE_URL:
        _vector_available = False
        return False

    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector';")).fetchone()
            _vector_available = bool(row)
    except Exception:
        _vector_available = False

    return _vector_available


# Enable pgvector extension when connecting to PostgreSQL if available
if "postgresql" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def enable_pgvector(dbapi_conn, connection_record):
        try:
            with dbapi_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            dbapi_conn.commit()
            global _vector_available
            _vector_available = True
        except Exception as e:
            try:
                dbapi_conn.rollback()
            except Exception:
                pass
            logger.warning(f"pgvector extension not installed in PostgreSQL or failed to enable: {e}")


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
    """Create all tables and columns if they do not exist."""
    import lead_fetcher.models  # ensure models are registered
    Base.metadata.create_all(bind=engine)

    # Ensure rating and review_count exist on leads table if leads already existed
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS rating FLOAT;"))
            conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS review_count INTEGER;"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Note on altering leads columns: {e}")
