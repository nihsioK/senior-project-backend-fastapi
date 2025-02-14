import argparse
import asyncio
import logging
from aiohttp import ClientSession
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    VideoStreamTrack
)
import cv2
from av import VideoFrame
from fractions import Fraction

logging.basicConfig(level=logging.INFO)

# For simplicity, use a public STUN server.
ICE_SERVERS = [
    RTCIceServer(
        urls=["stun:stun.l.google.com:19302"]
    )
]

RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)

class VideoCaptureTrack(VideoStreamTrack):
    """
    Captures frames from a local camera via OpenCV.
    """
    def __init__(self, device=0, fps=30):
        super().__init__()
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {device}")
        self.fps = fps
        self.time_base = Fraction(1, self.fps)
        self.next_pts = 0
        self.running = False

    async def recv(self):
        while not self.running:
            await asyncio.sleep(0.1)
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            logging.error("[Publisher] Failed to read frame from camera")
            raise RuntimeError("Failed to read frame from camera")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.next_pts
        video_frame.time_base = self.time_base
        self.next_pts += 1
        return video_frame

    async def start_stream(self):
        if not self.running:
            self.running = True
            logging.info("[Publisher] Streaming started.")

    async def stop_stream(self):
        if self.running:
            self.running = False
            logging.info("[Publisher] Streaming stopped.")

async def register_camera(device_id, server_url):
    async with ClientSession() as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/create/",
                json={"name": f"Camera_{device_id}", "camera_id": device_id}
            )
            if response.status in [200, 201]:
                logging.info(f"[Publisher] Camera '{device_id}' registered successfully.")
                return True
            else:
                logging.warning(f"[Publisher] Failed to register camera '{device_id}'. Status: {response.status}")
                return False
        except Exception as e:
            logging.error(f"[Publisher] Error registering camera: {e}")
            return False

async def set_connection(device_id, server_url, connection):
    async with ClientSession() as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/connect/",
                json={"camera_id": device_id, "connected": connection}
            )
            if response.status in [200, 201]:
                logging.info(f"[Publisher] Connection set to {connection} for camera '{device_id}'.")
                return True
            else:
                logging.warning(f"[Publisher] Failed to set connection for camera '{device_id}'. Status: {response.status}")
                return False
        except Exception as e:
            logging.error(f"[Publisher] Error setting connection: {e}")
            return False

# For simplicity, always keep the stream on.
async def control_stream(device_id, video_track, server_url):
    while True:
        if not video_track.running:
            await video_track.start_stream()
        await asyncio.sleep(10)

async def run(pc, session, cloud_server_url, camera_device, device_id):
    if not await register_camera(device_id, cloud_server_url):
        logging.error("[Publisher] Exiting: camera registration failed.")
        return
    if not await set_connection(device_id, cloud_server_url, True):
        logging.error("[Publisher] Exiting: connection setup failed.")
        return

    video_track = VideoCaptureTrack(device=camera_device)
    await video_track.start_stream()  # Ensure the track is active so SDP includes m=video
    pc.addTrack(video_track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    logging.info("[Publisher] Sending SDP Offer to server.")

    response = await session.post(
        f"{cloud_server_url}/offer",
        json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "role": "publisher",
            "device_id": device_id
        }
    )
    if response.status != 200:
        logging.error(f"[Publisher] Signaling failed with status {response.status}")
        return
    answer = await response.json()
    if answer.get("error"):
        logging.error(f"[Publisher] Error in publisher SDP answer: {answer['error']}")
        return
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    logging.info(f"[Publisher] Publisher connection established for device: {device_id}")

    asyncio.create_task(control_stream(device_id, video_track, cloud_server_url))

async def cleanup(device_id, server_url, pc):
    logging.info("[Publisher] Cleaning up before exit...")
    try:
        await set_connection(device_id, server_url, False)
    except Exception as e:
        logging.error(f"[Publisher] Cleanup failed: {e}")
    await pc.close()

async def main():
    parser = argparse.ArgumentParser(description="Publisher (Jetson) WebRTC Sender")
    parser.add_argument("--server", type=str, default="http://localhost:8082", help="Cloud server URL")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--device_id", type=str, required=True, help="Unique ID for this camera device")
    args = parser.parse_args()

    pc = RTCPeerConnection(RTC_CONFIG)

    async with ClientSession() as session:
        try:
            await run(pc, session, args.server, args.camera, args.device_id)
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await cleanup(args.device_id, args.server, pc)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Publisher] Interrupted by user.")
