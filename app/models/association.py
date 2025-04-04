from sqlalchemy import Table, Column, Integer, ForeignKey
from app.database import Base

user_camera_association = Table(
    "user_camera_association",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("camera_id", Integer, ForeignKey("cameras.id"), primary_key=True),
)
