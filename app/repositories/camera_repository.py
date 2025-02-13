from typing import Type

from sqlalchemy.orm import Session
from app.models.camera_models import Camera
from app.schemas.camera_schemas import CameraCreate

class CameraRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self) -> list[Type[Camera]]:
        return self.db.query(Camera).all()

    def get_by_camera_id(self, camera_id: str) -> Camera | None:
        return self.db.query(Camera).filter(Camera.camera_id == camera_id).first()

    def get_by_pk(self, pk: int) -> Camera | None:
        return self.db.query(Camera).filter(Camera.id == pk).first()

    def create(self, camera_create: CameraCreate) -> Camera:
        camera = self.get_by_camera_id(camera_create.camera_id)
        if camera:
            return camera

        camera = Camera(
            name=camera_create.name,
            camera_id=camera_create.camera_id,
        )
        self.db.add(camera)
        self.db.commit()
        self.db.refresh(camera)
        return camera

    def set_stream(self, camera_id: str, stream: bool) -> Camera | None:
        camera = self.get_by_camera_id(camera_id)
        if not camera:
            return None
        camera.stream = stream
        self.db.commit()
        self.db.refresh(camera)
        return camera

    def set_connection(self, camera_id: str, connected: bool) -> Camera | None:
        camera = self.get_by_camera_id(camera_id)
        if not camera:
            return None
        camera.connected = connected
        self.db.commit()
        self.db.refresh(camera)
        return camera

    def delete(self, camera_id: str) -> Camera | None:
        camera = self.get_by_camera_id(camera_id)
        if not camera:
            return None
        self.db.delete(camera)
        self.db.commit()
        return camera