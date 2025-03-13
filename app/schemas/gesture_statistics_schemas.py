from pydantic import BaseModel
from typing import List
from datetime import datetime


class ActionStatisticSchema(BaseModel):
    camera_id: str
    action: str
    count: int
    last_updated: datetime

    class Config:
        from_attributes = True
