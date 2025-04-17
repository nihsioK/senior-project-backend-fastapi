import argparse
import asyncio
import datetime
import json
import cv2
import numpy as np
import logging
from aiohttp import ClientSession, TCPConnector
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from av import VideoFrame
from fractions import Fraction
import websockets

from gesture_recognition import predict_from_frame

logging.basicConfig(level=logging.INFO, format="[Jetson] %(asctime)s - %(message)s")

ice_servers = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="turn:senior-backend.xyz:3478", username="testuser", credential="supersecretpassword"),
    RTCIceServer(urls="turns:senior-backend.xyz:5349", username="testuser", credential="supersecretpassword"),
]


latest_frame = None


class VideoCaptureTrack(VideoStreamTrack):
    def __init__(self, device=0, fps=15):
        super().__init__()
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {device}")

        self.fps = fps
        self.time_base = Fraction(1, self.fps)
        self.next_pts = 0
        self.running = False

    async def recv(self):
        global latest_frame
        while not self.running:
            await asyncio.sleep(0.1)

        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            logging.error("Failed to read frame from camera")
            raise RuntimeError("Camera read failure")

        latest_frame = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = self.next_pts
        video_frame.time_base = self.time_base
        self.next_pts += 1

        return video_frame

    async def start_stream(self):
        if not self.running:
            self.running = True
            logging.info("Streaming started.")

    async def stop_stream(self):
        if self.running:
            self.running = False
            logging.info("Streaming stopped.")


async def recognition_loop(device_id, ws_url, interval=0.5):
    global latest_frame
    retry_attempts = 0
    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                logging.info(f"Connected to WebSocket: {ws_url}")
                retry_attempts = 0  # reset after successful connect

                while True:
                    if latest_frame is None:
                        await asyncio.sleep(0.1)
                        continue

                    frame = latest_frame
                    action, confidence = predict_from_frame(frame)

                    if action != "I'm not sure":
                        payload = {
                            "camera_id": device_id,
                            "action": action,
                            "confidence": confidence,
                        }
                        await ws.send(json.dumps(payload))
                        logging.info(f"Sent recognition: {payload}")

                    if action == "rock":
                        alert_payload = {
                            "type": "alert",
                            "alert_level": "warning",
                            "camera_id": device_id,
                            "message": f"High confidence gesture: {action}",
                            "details": {
                                "action": action,
                                "confidence": confidence,
                                "timestamp": str(datetime.datetime.now()),
                            },
                        }
                        await ws.send(json.dumps(alert_payload))

                    await asyncio.sleep(interval)
        except Exception as e:
            logging.warning(f"WebSocket error: {e}")
            retry_attempts += 1
            await asyncio.sleep(min(2**retry_attempts, 30))  # exponential backoff


async def register_camera(device_id, server_url):
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/create/", json={"name": f"Camera_{device_id}", "camera_id": device_id}
            )
            return response.status in [200, 201]
        except Exception as e:
            logging.error(f"Camera registration failed: {e}")
            return False


async def set_connection(device_id, server_url, connected):
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/connect/", json={"camera_id": device_id, "connected": connected}
            )
            return response.status in [200, 201]
        except Exception as e:
            logging.error(f"Set connection failed: {e}")
            return False


async def set_stream(device_id, server_url, stream):
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/set_stream/", json={"camera_id": device_id, "stream": stream}
            )
            return response.status in [200, 201]
        except Exception as e:
            logging.error(f"Set stream failed: {e}")
            return False


async def control_stream(device_id, video_track, server_url):
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        while True:
            try:
                async with session.get(f"{server_url}/cameras/get/{device_id}") as resp:
                    cmd = await resp.json()
                    if cmd.get("stream"):
                        await video_track.start_stream()
                    else:
                        await video_track.stop_stream()
            except Exception as e:
                logging.warning(f"Polling stream state error: {e}")
            await asyncio.sleep(1)


async def run(pc, session, cloud_server_url, camera_device, device_id):
    pc.configuration = RTCConfiguration(iceServers=ice_servers)

    if not await register_camera(device_id, cloud_server_url):
        logging.error("Camera registration failed.")
        return

    if not await set_connection(device_id, cloud_server_url, True):
        logging.error("Connection setup failed.")
        return

    video_track = VideoCaptureTrack(device=camera_device)
    pc.addTrack(video_track)

    @pc.on("icecandidate")
    def on_ice_candidate(candidate):
        if candidate:
            logging.info(f"New ICE Candidate: {candidate}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        logging.info(f"ICE state: {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected"]:
            await pc.close()

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await asyncio.sleep(2)

    response = await session.post(
        f"{cloud_server_url}/offer",
        json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "role": "publisher",
            "device_id": device_id,
        },
    )

    if response.status != 200:
        logging.error(f"Offer failed: {response.status}")
        return

    answer = await response.json()
    if answer.get("error"):
        logging.error(f"SDP Error: {answer['error']}")
        return

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    logging.info("SDP handshake complete.")

    if not await set_stream(device_id, cloud_server_url, True):
        logging.error("Stream setup failed.")
        return

    asyncio.create_task(control_stream(device_id, video_track, cloud_server_url))


async def cleanup(device_id, cloud_server_url):
    logging.info("Cleaning up before exit...")
    try:
        await set_connection(device_id, cloud_server_url, False)
        await set_stream(device_id, cloud_server_url, False)
    except Exception as e:
        logging.warning(f"Cleanup error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Jetson Publisher with WebRTC + WebSocket")
    parser.add_argument("--server", type=str, default="https://senior-backend.xyz", help="Backend server URL")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--device_id", type=str, required=True, help="Unique camera ID")
    args = parser.parse_args()

    pc = RTCPeerConnection()
    server_url = args.server
    ws_protocol = "wss" if server_url.startswith("https") else "ws"
    ws_host = server_url.removeprefix("https://").removeprefix("http://")
    ws_url = f"{ws_protocol}://{ws_host}/ws/jetson"

    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        try:
            await run(pc, session, server_url, args.camera, args.device_id)
            asyncio.create_task(recognition_loop(args.device_id, ws_url))
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await cleanup(args.device_id, server_url)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
