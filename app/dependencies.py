from app.database import SessionLocal
import socket
import asyncio


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

publishers = {}
subscriber_pcs = set()

async def on_startup():
    """Startup event handler to initialize global settings."""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Server is running on IP: {local_ip}")
    except Exception as e:
        print("Could not obtain local IP:", e)

    try:
        from app.routers.webrtc import frame_processing_worker
        asyncio.create_task(frame_processing_worker())
    except Exception as e:
        print("Error starting frame processing worker:", e)

async def on_shutdown():
    """Shutdown event handler to clean up RTC connections."""
    for device_id, info in publishers.items():
        pc = info.get("pc")
        if pc:
            await pc.close()
    for pc in subscriber_pcs:
        await pc.close()
    print("All RTC connections closed.")
