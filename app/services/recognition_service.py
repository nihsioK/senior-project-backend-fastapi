from sqlalchemy.orm import Session
from typing import Optional, Type
from app.repositories.recognition_repository import RecognitionRepository
from app.schemas.recognition_schemas import RecognitionCreate, RecognitionOut
from app.models.recognition_models import HandGestureRecognition

class RecognitionService:
    def __init__(self, db: Session):
        self.recognition_repository = RecognitionRepository(db)

    def create_recognition(self, recognition_create: RecognitionCreate) -> HandGestureRecognition:
        return self.recognition_repository.create_or_update(recognition_create)

    def get_by_camera_id(self, camera_id: str) -> HandGestureRecognition | None:
        return self.recognition_repository.get_by_camera_id(camera_id)

    def get_all(self) -> list[Type[HandGestureRecognition]]:
        return self.recognition_repository.get_all()