from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.sql import func
from app.database import Base
import enum
from app.models.camera_models import Camera


class NotificationType(str, enum.Enum):
    CRITICAL = "critical"
    SEVERE = "severe"
    NORMAL = "normal"

class NotificationStatuses(str, enum.Enum):
    UNREAD = "unread"
    READ = "read"
    DELETED = "deleted"

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    type = Column(Enum(NotificationType), nullable=False, default=NotificationType.NORMAL)
    message = Column(String, nullable=False)
    camera_id = Column(String, nullable=False)
    status = Column(Enum(NotificationStatuses), nullable=False, default=NotificationStatuses.UNREAD)
