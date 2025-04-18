from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
from collections import defaultdict

router = APIRouter()

# Active connections
jetson_connections: Set[WebSocket] = set()
frontend_subscribers: Dict[str, Set[WebSocket]] = defaultdict(set)  # key = camera_id
alert_subscribers: Set[WebSocket] = set()


@router.websocket("/ws/jetson")
async def jetson_ws(websocket: WebSocket):
    await websocket.accept()
    jetson_connections.add(websocket)
    print("[WebSocket] Jetson connected")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                parsed = json.loads(data)
                camera_id = parsed.get("camera_id")
                if not camera_id:
                    continue  # Skip if no camera_id present

                # Check if this is an alert message
                if parsed.get("type") == "alert":
                    # Broadcast to all alert subscribers
                    disconnected = []
                    for client in alert_subscribers:
                        try:
                            await client.send_text(json.dumps(parsed))
                        except Exception as e:
                            print(f"[WebSocket] Failed to send alert: {e}")
                            disconnected.append(client)

                    # Clean up disconnected alert subscribers
                    for client in disconnected:
                        alert_subscribers.discard(client)

                # Regular recognition data (always process this part)
                for client in list(frontend_subscribers.get(camera_id, [])):
                    try:
                        await client.send_text(json.dumps(parsed))
                    except Exception as e:
                        print(f"[WebSocket] Failed to send to frontend ({camera_id}): {e}")
                        frontend_subscribers[camera_id].discard(client)
            except json.JSONDecodeError as e:
                print(f"[WebSocket] JSON decode error: {e}")
            except Exception as e:
                print(f"[WebSocket] Unexpected error in Jetson message handler: {e}")
    except WebSocketDisconnect:
        print("[WebSocket] Jetson disconnected")
    except Exception as e:
        print(f"[WebSocket] Jetson connection error: {e}")
    finally:
        jetson_connections.discard(websocket)


@router.websocket("/ws/frontend/{camera_id}")
async def frontend_ws(websocket: WebSocket, camera_id: str):
    """
    Frontend connects here to receive real-time recognition results from a specific camera.
    """
    await websocket.accept()
    frontend_subscribers[camera_id].add(websocket)
    print(f"[WebSocket] Frontend subscribed to camera: {camera_id}")
    try:
        while True:
            await websocket.receive_text()  # Optional: ping or keep-alive
    except WebSocketDisconnect:
        print(f"[WebSocket] Frontend disconnected from camera: {camera_id}")
    except Exception as e:
        print(f"[WebSocket] Frontend error: {e}")
    finally:
        frontend_subscribers[camera_id].discard(websocket)
        if not frontend_subscribers[camera_id]:
            del frontend_subscribers[camera_id]  # Clean up empty sets


@router.websocket("/ws/alert_system/")
async def alerts_ws(websocket: WebSocket):
    """
    Frontend clients connect here to receive alerts from any camera
    """
    await websocket.accept()
    alert_subscribers.add(websocket)
    print(f"[WebSocket] Client subscribed to alerts")
    try:
        while True:
            await websocket.receive_text()  # Keep-alive
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected from alerts")
    except Exception as e:
        print(f"[WebSocket] Alert subscriber error: {e}")
    finally:
        alert_subscribers.discard(websocket)
