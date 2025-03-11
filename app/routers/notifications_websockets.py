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
    This function runs as a background task and ensures only ONE listener is active.
    """
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("alerts")  # ✅ Subscribing to the 'alerts' channel

    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                alert_data = json.loads(message["data"])

                # Send alert to all connected WebSocket clients
                disconnected_clients = []
                for client_id, websocket in alert_connections.items():
                    try:
                        await websocket.send_json(alert_data)
                    except Exception as e:
                        print(f"[WebSocket Alert Error] {e}")
                        disconnected_clients.append(client_id)

                # Remove disconnected clients
                for client_id in disconnected_clients:
                    alert_connections.pop(client_id, None)
            except Exception as e:
                print(f"[WebSocket Alert Error] {e}")

# Start the alert listener once when FastAPI starts
asyncio.create_task(alert_listener())

@router.websocket("/ws/alerts/")
async def websocket_alert_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for clients to receive alert notifications.
    """
    await websocket.accept()  # ✅ Explicitly accept the WebSocket
    client_id = str(id(websocket))
    alert_connections[client_id] = websocket
    print(f"[WebSocket] Alert client connected: {client_id}")

    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        print(f"[WebSocket] Alert client disconnected: {client_id}")
        alert_connections.pop(client_id, None)

