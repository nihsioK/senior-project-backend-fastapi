from typing import Type

from sqlalchemy.orm import Session
from app.models.recognition_models import HandGestureRecognition
from app.schemas.recognition_schemas import RecognitionCreate

class RecognitionRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self) -> list[Type[HandGestureRecognition]]:
        return self.db.query(HandGestureRecognition).all()

    def get_by_camera_id(self, camera_id: str) -> HandGestureRecognition | None:
        return self.db.query(HandGestureRecognition).filter(HandGestureRecognition.camera_id == camera_id).first()

    def create_or_update(self, recognition_create: RecognitionCreate) -> HandGestureRecognition:
        recognition = self.get_by_camera_id(recognition_create.camera_id)

        if recognition:
            recognition.gesture = recognition_create.gesture
            self.db.commit()
            self.db.refresh(recognition)
        else:
            # Create a new recognition record
            recognition = HandGestureRecognition(
                camera_id=recognition_create.camera_id,
                gesture=recognition_create.gesture,
            )
            self.db.add(recognition)
            self.db.commit()
            self.db.refresh(recognition)

        return recognition
