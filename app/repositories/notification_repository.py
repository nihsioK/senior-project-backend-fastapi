from typing import Type

from sqlalchemy.orm import Session
from app.models.notification_models import Notification, NotificationType, NotificationStatuses
from app.schemas.notification_schemas import NotificationCreate

class NotificationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self) -> list[Type[Notification]]:
        return self.db.query(Notification).all()

    def get_by_pk(self, pk: int) -> Notification | None:
        return self.db.query(Notification).filter(Notification.id == pk).first()

    def get_by_camera_id(self, camera_id: str) -> list[Type[Notification]]:
        return self.db.query(Notification).filter(Notification.camera_id == camera_id).all()

    def create(self, notification_create: NotificationCreate) -> Notification:
        notification = Notification(
            type=notification_create.type,
            message=notification_create.message,
            camera_id=notification_create.camera_id,
        )
        self.db.add(notification)
        self.db.commit()
        self.db.refresh(notification)
        return notification

    def update(self, notification_id: int, status: NotificationStatuses) -> Notification | None:
        notification = self.get_by_pk(notification_id)
        if not notification:
            return None
        notification.status = status
        self.db.commit()
        self.db.refresh(notification)
        return notification

    def delete(self, notification_id: int) -> Notification | None:
        notification = self.get_by_pk(notification_id)
        if not notification:
            return None
        self.db.delete(notification)



