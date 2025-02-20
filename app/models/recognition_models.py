from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.sql import func
from app.database import Base
import enum

class HandGestureRecognition(Base):
    __tablename__ = "hand_gesture_recognition"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    gesture = Column(String, nullable=False)