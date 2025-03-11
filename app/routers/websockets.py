import aioredis
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict

router = APIRouter()
redis_client = aioredis.from_url("redis://localhost", decode_responses=True)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


async def redis_listener(websocket: WebSocket, device_id: str):
    """
    Asynchronously listens to Redis Pub/Sub and sends action results to WebSocket clients.
    """
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("action_results")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                action_data = json.loads(message["data"])

                if action_data["device_id"] == device_id and device_id in active_connections:
                    await active_connections[device_id].send_json(action_data)
    except Exception as e:
        print(f"[WebSocket Error] {e}")
    finally:
        if device_id in active_connections:
            del active_connections[device_id]


@router.websocket("/ws/actions/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    """
    WebSocket endpoint for clients to receive action recognition results.
    """
    await websocket.accept()
    active_connections[device_id] = websocket
    print(f"[WebSocket] Client connected for device: {device_id}")

    try:
        await redis_listener(websocket, device_id)
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for device: {device_id}")
        active_connections.pop(device_id, None)
