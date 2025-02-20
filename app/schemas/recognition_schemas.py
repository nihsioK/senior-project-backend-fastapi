from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class RecognitionBase(BaseModel):
    camera_id: str
    gesture: str

class RecognitionCreate(RecognitionBase):
    pass

class RecognitionOut(RecognitionBase):
    id: int
    created_at: datetime