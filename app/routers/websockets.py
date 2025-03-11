import json
import asyncio
import redis.asyncio as redis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from typing import Dict

router = APIRouter()

# Initialize Async Redis client
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


async def redis_listener(device_id: str):
    """
    Listens to Redis Pub/Sub asynchronously and sends action results to WebSocket clients.
    """
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("action_results")  # ✅ Use async subscribe

    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                action_data = json.loads(message["data"])

                # Ensure device is connected before sending data
                if action_data["device_id"] == device_id and device_id in active_connections:
                    websocket = active_connections[device_id]
                    await websocket.send_json(action_data)  # ✅ Await WebSocket message
            except Exception as e:
                print(f"[WebSocket Error] {e}")


@router.websocket("/ws/actions/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    """
    WebSocket endpoint for clients to receive action recognition results.
    """
    await websocket.accept()
    active_connections[device_id] = websocket
    print(f"[WebSocket] Client connected for device: {device_id}")

    # ✅ Run the Redis listener in a background task (non-blocking)
    asyncio.create_task(redis_listener(device_id))

    try:
        while True:
            await websocket.receive_text()  # Keep WebSocket connection alive
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for device: {device_id}")
        active_connections.pop(device_id, None)
