from sqlalchemy.orm import Session
from typing import Optional
from app.repositories.user_repository import UserRepository
from app.schemas.user_schemas import UserCreate
from app.models.user_models import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserService:
    def __init__(self, db: Session):
        self.user_repository = UserRepository(db)

    def register_user(self, user_create: UserCreate) -> User:
        if self.user_repository.get_by_email(user_create.email):
            raise ValueError("Email already registered")
        return self.user_repository.create(user_create)

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        user = self.user_repository.get_by_email(email)
        if not user or not pwd_context.verify(password, user.hashed_password):
            return None
        return user
