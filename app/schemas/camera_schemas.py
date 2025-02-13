from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class CameraBase(BaseModel):
    name: str
    camera_id: str

class CameraCreate(CameraBase):
    pass

class CameraOut(CameraBase):
    id: int
    stream: bool
    created_at: datetime
    connected: bool

    class Config:
        from_attributes = True

class CameraSetStream(BaseModel):
    camera_id: str
    stream: bool

class CameraSetConnection(BaseModel):
    camera_id: str
    connected: bool