from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta
from app.auth.authentication import create_access_token
from app.schemas.recognition_schemas import RecognitionCreate, RecognitionOut
from app.services.recognition_service import RecognitionService
from app.dependencies import get_db

router = APIRouter(
    prefix="/recognitions",
    tags=["recognitions"]
)

@router.post("/create", response_model=RecognitionOut)
def create_recognition(recognition: RecognitionCreate, db: Session = Depends(get_db)):
    recognition_service = RecognitionService(db)
    try:
        recognition = recognition_service.create_recognition(recognition)
        return recognition
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/get_all", response_model=list[RecognitionOut])
def get_all(db: Session = Depends(get_db)):
    recognition_service = RecognitionService(db)
    recognitions = recognition_service.get_all()
    return recognitions

@router.get("/get_by_id/{recognition_id}", response_model=RecognitionOut)
def get_by_id(recognition_id: str, db: Session = Depends(get_db)):
    recognition_service = RecognitionService(db)
    recognition = recognition_service.get_by_camera_id(recognition_id)
    if not recognition:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recognition not found")
    return recognition