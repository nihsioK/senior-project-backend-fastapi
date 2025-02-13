from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta
from app.auth.authentication import create_access_token
from app.schemas.camera_schemas import CameraCreate, CameraOut, CameraSetStream, CameraSetConnection
from app.services.camera_service import CameraService
from app.dependencies import get_db

router = APIRouter(
    prefix="/cameras",
    tags=["cameras"]
)

@router.post("/create", response_model=CameraOut)
def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    camera_service = CameraService(db)
    try:
        camera = camera_service.create_camera(camera)
        return camera
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/set_stream", response_model=CameraOut)
def set_stream(camera: CameraSetStream, db: Session = Depends(get_db)):
    camera_service = CameraService(db)
    try:
        camera = camera_service.set_stream(camera.camera_id, camera.stream)
        return camera
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/get/{camera_id}", response_model=CameraOut)
def get_by_camera_id(camera_id: str, db: Session = Depends(get_db)):
    camera_service = CameraService(db)
    camera = camera_service.get_by_camera_id(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return camera

@router.get("/get_all", response_model=list[CameraOut])
def get_all(db: Session = Depends(get_db)):
    camera_service = CameraService(db)
    cameras = camera_service.get_all()
    return cameras

@router.post("/connect", response_model=CameraOut)
def connect(camera: CameraSetConnection, db: Session = Depends(get_db)):
    camera_service = CameraService(db)
    try:
        camera = camera_service.set_connection(camera.camera_id, camera.connected)
        return camera
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/delete/{camera_id}", response_model=CameraOut)
def delete(camera_id: str, db: Session = Depends(get_db)):
    camera_service = CameraService(db)
    camera = camera_service.delete(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return camera