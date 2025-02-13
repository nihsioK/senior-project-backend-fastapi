from sqlalchemy.orm import Session
from typing import Optional, Type
from app.repositories.camera_repository import CameraRepository
from app.schemas.camera_schemas import CameraCreate, CameraOut
from app.models.camera_models import Camera
from app.dependencies import publishers

class CameraService:
    def __init__(self, db: Session):
        self.camera_repository = CameraRepository(db)

    def create_camera(self, camera_create: CameraCreate) -> Camera:
        return self.camera_repository.create(camera_create)

    def set_stream(self, camera_id: str, stream: bool) -> Camera | None:
        camera = self.camera_repository.set_stream(camera_id, stream)
        if not camera:
            return None

        if camera_id in publishers:
            publishers[camera_id]["streaming"] = stream

        return camera

    def set_connection(self, camera_id: str, connected: bool) -> Camera | None:
        camera = self.camera_repository.set_connection(camera_id, connected)
        if not camera:
            return None
        return camera

    def get_by_camera_id(self, camera_id: str) -> Camera | None:
        return self.camera_repository.get_by_camera_id(camera_id)

    def get_by_pk(self, pk: int) -> Camera | None:
        return self.camera_repository.get_by_pk(pk)

    def get_all(self) -> list[Type[Camera]]:
        return self.camera_repository.get_all()

    def delete(self, camera_id: str) -> Camera | None:
        return self.camera_repository.delete(camera_id)