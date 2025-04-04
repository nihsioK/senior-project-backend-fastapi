from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.auth.authentication import create_access_token
from app.schemas.camera_schemas import CameraOut
from app.schemas.user_schemas import UserCreate, UserOut, Token
from app.services.user_service import UserService
from app.dependencies import get_db
from app.config import ACCESS_TOKEN_EXPIRE_MINUTES
from app.auth.authentication import get_current_user
from app.models.user_models import User

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.post("/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    user_service = UserService(db)
    try:
        new_user = user_service.register_user(user)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return new_user

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user_service = UserService(db)
    user = user_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/{user_id}/cameras", response_model=list[CameraOut])
def get_user_cameras(user_id: int, db: Session = Depends(get_db)):
    user_service = UserService(db)
    return user_service.get_user_cameras(user_id)

@router.get("/me/cameras/new", response_model=list[CameraOut] )
def get_my_cameras(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_service = UserService(db)
    cameras = user_service.get_user_cameras(current_user.id)
    return user_service.get_user_cameras(current_user.id)

@router.get("/me/cameras/old")
def get_my_cameras_old(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_service = UserService(db)
    return user_service.get_user_cameras(current_user.id)

@router.post("/{user_id}/cameras/{camera_id}/link")
def link_camera(user_id: int, camera_id: int, db: Session = Depends(get_db)):
    user_service = UserService(db)
    success = user_service.link_camera_to_user(user_id, camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="User or camera not found")
    return {"message": "Camera linked successfully"}

@router.delete("/{user_id}/cameras/{camera_id}/unlink")
def unlink_camera(user_id: int, camera_id: int, db: Session = Depends(get_db)):
    user_service = UserService(db)
    success = user_service.unlink_camera_from_user(user_id, camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="User or camera not found or not linked")
    return {"message": "Camera unlinked successfully"}


@router.get("/list")
def list_users(db: Session = Depends(get_db)):
    user_service = UserService(db)
    return user_service.get_all_users()