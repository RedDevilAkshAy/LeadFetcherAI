from sqlalchemy import Column, Integer, String

try:
    from database import Base
except ImportError:
    from ..database import Base


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True)
    business_name = Column(String(100), nullable=True)
    business_website = Column(String(100), nullable=True)
    owner_name = Column(String(50), nullable=True)
    email = Column(String(255), unique=True, nullable=True)
    number = Column(String(20), nullable=True)
    characteristic = Column(String(20), nullable=True)
    location = Column(String(100), nullable=True)
