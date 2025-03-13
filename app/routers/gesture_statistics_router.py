from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from app.dependencies import get_db
from app.schemas.gesture_statistics_schemas import ActionStatisticSchema
from app.services.gesture_statistics_service import ActionStatisticService
from typing import List

router = APIRouter(prefix="/actions", tags=["actions"])


@router.post("/process", response_model=ActionStatisticSchema)
def process_action(camera_id: str, action: str, db: Session = Depends(get_db)):
    action_service = ActionStatisticService()
    try:
        result = action_service.process_action(db, camera_id, action)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/statistics/{camera_id}", response_model=List[ActionStatisticSchema])
def get_statistics(camera_id: str, db: Session = Depends(get_db)):
    action_service = ActionStatisticService()
    stats = action_service.get_camera_statistics(db, camera_id)
    if not stats:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No statistics found for this camera")
    return stats
