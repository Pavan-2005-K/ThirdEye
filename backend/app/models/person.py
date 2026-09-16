from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from app.database.database import Base


class Person(Base):
    __tablename__ = "registered_people"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(150), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(50), nullable=True)

    address = Column(String(500), nullable=True)
    phone = Column(String(30), nullable=True)

    photo_path = Column(String(500), nullable=False)
    embedding_path = Column(String(500), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )