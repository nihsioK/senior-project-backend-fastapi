from sqlalchemy.orm import Session
from typing import Optional, Type
from app.repositories.notification_repository import NotificationRepository
from app.schemas.notification_schemas import NotificationCreate
from app.models.notification_models import Notification, NotificationType, NotificationStatuses

class NotificationService:
    def __init__(self, db: Session):
        self.notification_repository = NotificationRepository(db)

    def create_notification(self, notification_create: NotificationCreate) -> Notification:
        return self.notification_repository.create(notification_create)

    def get_by_pk(self, pk: int) -> Notification | None:
        return self.notification_repository.get_by_pk(pk)

    def get_by_camera_id(self, camera_id: str) -> list[Type[Notification]]:
        return self.notification_repository.get_by_camera_id(camera_id)

    def get_all(self) -> list[Type[Notification]]:
        return self.notification_repository.get_all()

    def delete(self, notification_id: int) -> Notification | None:
        return self.notification_repository.delete(notification_id)

    def update(self, notification_id: int, status: NotificationStatuses) -> Notification | None:
        return self.notification_repository.update(notification_id, status)
