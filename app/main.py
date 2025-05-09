from fastapi import FastAPI
from app.database import engine, Base
from app.routers import (
    user_router,
    camera_router,
    webrtc,
    notification_router,
    recognition_router,
    websockets,
    notifications_websockets,
    gesture_statistics_router,
    websockets_recognition,
)
from fastapi.middleware.cors import CORSMiddleware
from app.dependencies import on_startup, on_shutdown
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Base.metadata.create_all(bind=engine)

app = FastAPI(title="Senior Project")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

origins = [
    "http://localhost:3000",
    "https://example.com",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_event_handler("startup", on_startup)
app.add_event_handler("shutdown", on_shutdown)

app.include_router(user_router.router)
app.include_router(camera_router.router)
app.include_router(webrtc.router)

app.include_router(gesture_statistics_router.router)
app.include_router(websockets.router)
# app.include_router(notifications_websockets.router)

app.include_router(notification_router.router)

app.include_router(recognition_router.router)

app.include_router(websockets_recognition.router)


@app.get("/")
async def index():
    with open("app/static/index.html") as f:
        return HTMLResponse(f.read())
