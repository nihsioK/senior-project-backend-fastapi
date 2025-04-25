from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
from collections import defaultdict, deque
from fastapi import Depends
from sqlalchemy.orm import Session
import logging
from app.dependencies import get_db
from app.services.gesture_statistics_service import ActionStatisticService
from app.repositories.notification_repository import NotificationRepository
from app.models.notification_models import NotificationType
from fastapi.concurrency import run_in_threadpool
import asyncio
from app.database import SessionLocal

router = APIRouter()

logger = logging.getLogger(__name__)


# Active connections
jetson_connections: Set[WebSocket] = set()
frontend_subscribers: Dict[str, Set[WebSocket]] = defaultdict(set)  # key = camera_id
alert_subscribers: Set[WebSocket] = set()

_ACTION_BUFFER: deque[tuple[str, str]] = deque()
_FLUSH_SIZE = 50
_FLUSH_DELAY = 0.20  # or after 200 ms of silence
_flush_timer: asyncio.Task | None = None  # global handle


action_service = ActionStatisticService()  # share one instance


def _write_actions_sync(batch: list[tuple[str, str]]) -> None:
    """
    Runs inside a worker thread; uses a *blocking* sync SQLAlchemy session.
    """
    db = SessionLocal()
    try:
        for cam_id, gesture in batch:
            action_service.process_action(None, cam_id, gesture)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def _flush_actions_if_any() -> None:
    """
    Pops everything from _ACTION_BUFFER and commits in one go.
    Must be called from the event loop.
    """
    global _ACTION_BUFFER
    if not _ACTION_BUFFER:
        return

    batch: list[tuple[str, str]] = []
    while _ACTION_BUFFER:
        batch.append(_ACTION_BUFFER.popleft())

    try:
        await run_in_threadpool(_write_actions_sync, batch)
        logger.debug(f"Flushed {len(batch)} actions.")
    except Exception as e:
        logger.error(f"[Flush] failed ({len(batch)} rows): {e}")


async def _schedule_delayed_flush():
    """
    Waits FLUSH_DELAY seconds, then flushes. Re-scheduled on every new record.
    """
    global _flush_timer
    try:
        await asyncio.sleep(_FLUSH_DELAY)
        await _flush_actions_if_any()
    finally:
        _flush_timer = None


def _kick_timer() -> None:
    """
    Ensures exactly one delayed-flush task is pending at any moment.
    Called after each buffer insertion when buffer < FLUSH_SIZE.
    """
    global _flush_timer
    if _flush_timer is None or _flush_timer.done():
        _flush_timer = asyncio.create_task(_schedule_delayed_flush())


@router.websocket("/ws/jetson")
async def jetson_ws(
    websocket: WebSocket,
    db: Session = Depends(get_db),  # for ALERT writes only
):
    await websocket.accept()
    jetson_connections.add(websocket)
    notification_repo = NotificationRepository(db)
    logger.info("[Jetson] connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
                cam_id: str | None = event.get("camera_id")
                gesture: str | None = event.get("action")
                msg_type: str | None = event.get("type")

                if not cam_id:
                    continue

                # 1) ---------- normal recognitions (buffered) ----------------
                if msg_type != "alert" and gesture:
                    _ACTION_BUFFER.append((cam_id, gesture))

                    if len(_ACTION_BUFFER) >= _FLUSH_SIZE:
                        # cancel pending timer & flush now
                        if _flush_timer:
                            _flush_timer.cancel()
                        await _flush_actions_if_any()
                    else:
                        _kick_timer()

                # 2) ---------- alerts (immediate) ----------------------------
                if msg_type == "alert":

                    def _write_alert_sync():
                        notification_repo.create_manual(
                            notification_type=NotificationType.CRITICAL,
                            message=f"On {cam_id} detected unusual activity: {gesture}",
                            camera_id=cam_id,
                        )
                        db.commit()

                    await run_in_threadpool(_write_alert_sync)

                    # broadcast alert
                    disconnected = []
                    for client in alert_subscribers:
                        try:
                            await client.send_text(raw)
                        except Exception:
                            disconnected.append(client)
                    for c in disconnected:
                        alert_subscribers.discard(c)

                # 3) ---------- fan-out to dashboards -------------------------
                for client in list(frontend_subscribers.get(cam_id, [])):
                    try:
                        await client.send_text(raw)
                    except Exception:
                        frontend_subscribers[cam_id].discard(client)

            except json.JSONDecodeError:
                logger.warning("[Jetson] bad JSON")
            except Exception as e:
                logger.error(f"[Jetson] handler error: {e}")

    except WebSocketDisconnect:
        logger.info("[Jetson] disconnected")
    finally:
        jetson_connections.discard(websocket)
        # best-effort final flush
        await _flush_actions_if_any()
        db.close()


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
