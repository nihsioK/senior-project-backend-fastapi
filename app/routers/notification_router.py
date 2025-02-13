from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from app.models.notification_models import NotificationStatuses
from app.schemas.notification_schemas import NotificationStatus, NotificationCreate, NotificationResponse, NotificationType
from app.services.notification_service import NotificationService
from app.dependencies import get_db

router = APIRouter(
    prefix="/notifications",
    tags=["notifications"]
)

@router.post("/create", response_model=NotificationResponse)
def create_notification(notification: NotificationCreate, db: Session = Depends(get_db)):
    notification_service = NotificationService(db)
    try:
        notification = notification_service.create_notification(notification)
        return notification
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/get_all", response_model=list[NotificationResponse])
def get_all(db: Session = Depends(get_db)):
    notification_service = NotificationService(db)
    notifications = notification_service.get_all()
    return notifications


@router.get("/get_by_id/{notification_id}", response_model=NotificationResponse)
def get_by_id(notification_id: int, db: Session = Depends(get_db)):
    notification_service = NotificationService(db)
    notification = notification_service.get_by_pk(notification_id)
    if not notification:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    return notification


@router.get("/get_by_camera_id/{camera_id}", response_model=list[NotificationResponse])
def get_by_camera_id(camera_id: str, db: Session = Depends(get_db)):
    notification_service = NotificationService(db)
    notifications = notification_service.get_by_camera_id(camera_id)
    return notifications


@router.post("/update_status/{notification_id}", response_model=NotificationResponse)
def update_status(notification_id: int, statuses: NotificationStatuses, db: Session = Depends(get_db)):
    notification_service = NotificationService(db)
    notification = notification_service.update(notification_id, statuses)
    if not notification:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    return notification