from sqlalchemy.orm import Session

from app.models.camera_models import Camera
from app.models.user_models import User
from app.schemas.user_schemas import UserCreate
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str):
        return self.db.query(User).filter(User.email == email).first()

    def create(self, user_create: UserCreate) -> User:
        hashed_password = pwd_context.hash(user_create.password)
        user = User(
            name=user_create.name,
            email=user_create.email,
            hashed_password=hashed_password
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_user_cameras(self, user_id: int) -> list[Camera]:
        user = self.db.query(User).filter(User.id == user_id).first()
        if user:
            return user.cameras
        return []

    def link_camera(self, user_id: int, camera_id: int) -> bool:
        user = self.db.query(User).filter(User.id == user_id).first()
        camera = self.db.query(Camera).filter(Camera.id == camera_id).first()

        if not user or not camera:
            return False  # Or raise an exception

        if camera not in user.cameras:
            user.cameras.append(camera)
            self.db.commit()

        return True

    def unlink_camera(self, user_id: int, camera_id: int) -> bool:
        user = self.db.query(User).filter(User.id == user_id).first()
        camera = self.db.query(Camera).filter(Camera.id == camera_id).first()

        if not user or not camera:
            return False  # Or raise an exception

        if camera in user.cameras:
            user.cameras.remove(camera)
            self.db.commit()

        return True
