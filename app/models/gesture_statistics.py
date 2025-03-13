from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base
from sqlalchemy.sql import func


class ActionStatistic(Base):
    __tablename__ = "action_statistics"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.camera_id"), nullable=False)
    action = Column(String, nullable=False)
    count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())

    camera = relationship("Camera", back_populates="statistics")

    __table_args__ = (UniqueConstraint("camera_id", "action", name="uq_camera_action"),)
