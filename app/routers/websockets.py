import json
import redis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from typing import Dict

router = APIRouter()

# Initialize Redis client
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


def redis_listener(websocket: WebSocket, device_id: str):
    """
    Listens to Redis Pub/Sub and sends action results to WebSocket clients.
    """
    pubsub = redis_client.pubsub()
    pubsub.subscribe("action_results")  # Remove 'await' (sync call)

    try:
        for message in pubsub.listen():
            if message["type"] == "message":
                action_data = json.loads(message["data"])

                if action_data["device_id"] == device_id and device_id in active_connections:
                    # Send WebSocket message inside an async task
                    websocket.send_json(action_data)
    except Exception as e:
        print(f"[WebSocket Error] {e}")
    finally:
        if device_id in active_connections:
            del active_connections[device_id]


@router.websocket("/ws/actions/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str, background_tasks: BackgroundTasks):
    """
    WebSocket endpoint for clients to receive action recognition results.
    """
    await websocket.accept()
    active_connections[device_id] = websocket
    print(f"[WebSocket] Client connected for device: {device_id}")

    # Run the Redis listener as a background task (non-blocking)
    background_tasks.add_task(redis_listener, websocket, device_id)

    try:
        while True:
            await websocket.receive_text()  # Keep WebSocket connection alive
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for device: {device_id}")
        active_connections.pop(device_id, None)
