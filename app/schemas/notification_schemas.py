from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.models.notification_models import NotificationType, NotificationStatuses

class NotificationBase(BaseModel):
    type: NotificationType
    message: str
    camera_id: Optional[str] = None

    class Config:
        from_attributes = True


class NotificationCreate(NotificationBase):
    pass


class NotificationResponse(NotificationBase):
    id: int
    created_at: datetime
    status: NotificationStatuses

class NotificationStatus(BaseModel):
    status: NotificationStatuses
