from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.database import Base
from sqlalchemy.orm import relationship


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    camera_id = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    stream = Column(Boolean, default=False)
    connected = Column(Boolean, default=False)

    statistics = relationship("ActionStatistic", back_populates="camera", cascade="all, delete-orphan")
