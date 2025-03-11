import json
import asyncio
import redis.asyncio as redis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict

router = APIRouter()

# Initialize Async Redis client
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Store active WebSocket connections for alerts
alert_connections: Dict[str, WebSocket] = {}


async def alert_listener():
    """
    Listens for alerts in Redis Pub/Sub and sends notifications to connected WebSocket clients.
    """
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("alerts")  # ✅ Subscribing to the 'alerts' channel

    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                alert_data = json.loads(message["data"])

                # Send alert to all connected clients
                for websocket in alert_connections.values():
                    await websocket.send_json(alert_data)  # ✅ Sending alert to frontend
            except Exception as e:
                print(f"[WebSocket Alert Error] {e}")


@router.websocket("/ws/alerts")
async def websocket_alert_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for clients to receive alert notifications.
    """
    await websocket.accept()
    client_id = str(id(websocket))  # Unique identifier for each connection
    alert_connections[client_id] = websocket
    print(f"[WebSocket] Alert client connected: {client_id}")

    # ✅ Run the alert listener in a background task
    asyncio.create_task(alert_listener())

    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        print(f"[WebSocket] Alert client disconnected: {client_id}")
        alert_connections.pop(client_id, None)
