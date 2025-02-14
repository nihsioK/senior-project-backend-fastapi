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

# Optional: enable debug logging for aiortc
logging.basicConfig(level=logging.INFO)

# TURN server configuration
ICE_SERVERS = [
    RTCIceServer(
        urls=["turn:46.8.31.7:3478?transport=udp", "turn:46.8.31.7:3478?transport=tcp"],
        username="webrtcuser",
        credential="strongpassword",
        credentialType="password"
    )
]

RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)

class VideoCaptureTrack(VideoStreamTrack):
    """
    A VideoStreamTrack that captures frames from a local camera using OpenCV
    and provides them to aiortc for sending over WebRTC.
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
        """Grabs a frame from the camera and returns it as a VideoFrame."""
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
        """Set streaming to True, so frames will be read."""
        if not self.running:
            self.running = True
            logging.info("[Publisher] Streaming started.")

    async def stop_stream(self):
        """Set streaming to False, so frames will not be read."""
        if self.running:
            self.running = False
            logging.info("[Publisher] Streaming stopped.")


async def register_camera(device_id, server_url):
    """
    Example endpoint call: /cameras/create/
    Registers the camera (device_id) with the server.
    """
    async with ClientSession() as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/create/",
                json={
                    "name": f"Camera_{device_id}",
                    "camera_id": device_id
                }
            )
            if response.status in [200, 201]:
                logging.info(f"[Publisher] Camera '{device_id}' registered successfully.")
                return True
            else:
                logging.warning(
                    f"[Publisher] Failed to register camera '{device_id}'. "
                    f"Status: {response.status}"
                )
                return False
        except Exception as e:
            logging.error(f"[Publisher] Error registering camera: {e}")
            return False


async def set_connection(device_id, server_url, connection):
    """
    Example endpoint call: /cameras/connect/
    Tells the server we are connected or disconnected.
    """
    async with ClientSession() as session:
        try:
            response = await session.post(
                f"{server_url}/cameras/connect/",
                json={
                    "camera_id": device_id,
                    "connected": connection
                }
            )
            if response.status in [200, 201]:
                logging.info(
                    f"[Publisher] Connection set to {connection} for camera '{device_id}'."
                )
                return True
            else:
                logging.warning(
                    f"[Publisher] Failed to set connection for camera '{device_id}'. "
                    f"Status: {response.status}"
                )
        except Exception as e:
            logging.error(f"[Publisher] Error setting connection: {e}")
            return False


async def control_stream(device_id, video_track, server_url):
    """
    Periodically polls the server to see if 'stream' is True or False,
    then starts or stops the local video capture accordingly.
    """
    async with ClientSession() as session:
        while True:
            try:
                async with session.get(f"{server_url}/cameras/get/{device_id}") as resp:
                    cmd = await resp.json()
                    if cmd.get("stream"):
                        await video_track.start_stream()
                    else:
                        await video_track.stop_stream()
            except Exception as e:
                logging.error(f"[Publisher] Error polling device_command: {e}")
            await asyncio.sleep(1)


async def run(pc, session, cloud_server_url, camera_device, device_id):
    """
    Main logic for:
    1) Register camera
    2) Announce connection
    3) Create local video track
    4) Send offer to server
    5) Handle answer
    6) Start a task to poll for streaming on/off
    """
    # Step 1: Register camera
    if not await register_camera(device_id, cloud_server_url):
        logging.error("[Publisher] Exiting due to camera registration failure.")
        return

    # Step 2: Set connection = True
    if not await set_connection(device_id, cloud_server_url, True):
        logging.error("[Publisher] Exiting due to connection setup failure.")
        return

    # Step 3: Create local video track, add to PeerConnection
    video_track = VideoCaptureTrack(device=camera_device)
    pc.addTrack(video_track)

    # Step 4: Create offer, setLocalDescription
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    logging.info("[Publisher] Sending SDP Offer to server.")

    # Step 5: POST offer to /offer
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

    # Step 6: setRemoteDescription with the server's answer
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )
    logging.info(f"[Publisher] Publisher connection established for device: {device_id}")

    # Step 7: Start background task that polls the server for streaming status
    asyncio.create_task(control_stream(device_id, video_track, cloud_server_url))


async def cleanup(device_id, cloud_server_url):
    """
    Called upon exit to mark the camera as disconnected on the server.
    """
    logging.info("[Publisher] Cleaning up before exit...")
    try:
        await set_connection(device_id, cloud_server_url, False)
    except Exception as e:
        logging.error(f"[Publisher] Cleanup failed: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Publisher (Jetson) WebRTC Sender")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8082",
        help="Cloud server URL"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index"
    )
    parser.add_argument(
        "--device_id",
        type=str,
        required=True,
        help="Unique ID for this camera device"
    )
    args = parser.parse_args()

    # Create RTCPeerConnection with TURN config
    pc = RTCPeerConnection(RTC_CONFIG)

    async with ClientSession() as session:
        try:
            await run(pc, session, args.server, args.camera, args.device_id)
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await cleanup(args.device_id, args.server)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Publisher] Interrupted by user.")
