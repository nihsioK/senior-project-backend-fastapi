import argparse
import asyncio
from aiohttp import ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
import cv2
from av import VideoFrame
from fractions import Fraction
import requests

API_KEY = "2492bffdb915ca4e706d051ea6bb8de323ff"

def get_turn_credentials():
    url = f"https://senior.metered.live/api/v1/turn/credentials?apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        turn_servers = response.json()
        return [
            RTCIceServer(
                urls=server["urls"],  # Keep only TCP transport
                username=server.get("username", ""),
                credential=server.get("credential", "")
            )
            for server in turn_servers if "transport=tcp" in server["urls"]
        ]
    else:
        print(f"Failed to fetch TURN credentials: {response.status_code}")
        return []


ice_servers = [
    RTCIceServer(urls="turn:46.8.31.7:3478", username="testuser", credential="supersecretpassword"),
    RTCIceServer(urls="turns:46.8.31.7:5349", username="testuser", credential="supersecretpassword")
]


class VideoCaptureTrack(VideoStreamTrack):
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
        self.running = False  # Do not capture frames until commanded.

    async def recv(self):
        while not self.running:
            await asyncio.sleep(0.1)
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            print("[Publisher] ERROR: Failed to read frame from camera")
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
            print("[Publisher] Streaming started.")

    async def stop_stream(self):
        if self.running:
            self.running = False
            print("[Publisher] Streaming stopped.")


async def register_camera(device_id, server_url):
    async with ClientSession() as session:
        try:
            response = await session.post(f"{server_url}/cameras/create/", json={
                "name": f"Camera_{device_id}",
                "camera_id": device_id
            })
            if response.status in [200, 201]:
                print(f"[Publisher] Camera '{device_id}' registered successfully.")
                return True
            else:
                print(f"[Publisher] Failed to register camera '{device_id}'. Status: {response.status}")
                return False
        except Exception as e:
            print(f"[Publisher] Error registering camera: {e}")
            return False


async def set_connection(device_id, server_url, connection):
    async with ClientSession() as session:
        try:
            response = await session.post(f"{server_url}/cameras/connect/", json={
                "camera_id": device_id,
                "connected": connection
            })
            if response.status in [200, 201]:
                print(f"[Publisher] Connection set to {connection} for camera '{device_id}'.")
                return True
            else:
                print(f"[Publisher] Failed to set connection for camera '{device_id}'. Status: {response.status}")
        except Exception as e:
            print(f"[Publisher] Error setting connection: {e}")
            return False


async def control_stream(device_id, video_track, server_url):
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
                print(f"[Publisher] Error polling device_command: {e}")
            await asyncio.sleep(1)


async def run(pc, session, cloud_server_url, camera_device, device_id):
    # Configure ICE servers
    pc.configuration = RTCConfiguration(iceServers=ice_servers)


    if not await register_camera(device_id, cloud_server_url):
        print("[Publisher] Exiting due to camera registration failure.")
        return

    if not await set_connection(device_id, cloud_server_url, True):
        print("[Publisher] Exiting due to connection setup failure.")
        return

    video_track = VideoCaptureTrack(device=camera_device)
    pc.addTrack(video_track)

    # Add ICE candidate handling
    @pc.on("icecandidate")
    def on_ice_candidate(candidate):
        if candidate:
            print(f"[Publisher] New ICE Candidate: {candidate}")
        else:
            print("[Publisher] ICE Candidate gathering finished.")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        print(f"[Publisher] ICE Connection State: {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected"]:
            print("[Publisher] ICE Connection failed. Attempting reconnection...")
            # Implement reconnection logic here

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    await asyncio.sleep(2)

    print("[Publisher] Sending SDP Offer:", offer)
    response = await session.post(f"{cloud_server_url}/offer", json={
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "role": "publisher",
        "device_id": device_id
    })

    if response.status != 200:
        print(f"[Publisher] Signaling failed with status {response.status}")
        print(f"Error details: ", await response.text())
        return

    answer = await response.json()
    if answer.get("error"):
        print(f"[Publisher] Error in publisher SDP answer: {answer['error']}")
        return

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    print("[Publisher] Publisher connection established for device:", device_id)

    asyncio.create_task(control_stream(device_id, video_track, cloud_server_url))


async def cleanup(device_id, cloud_server_url):
    print("[Publisher] Cleaning up before exit...")
    try:
        await set_connection(device_id, cloud_server_url, False)
    except Exception as e:
        print(f"[Publisher] Cleanup failed: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Publisher (Jetson) WebRTC Sender")
    parser.add_argument("--server", type=str, default="http://localhost:8082", help="Cloud server URL")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--device_id", type=str, required=True, help="Unique ID for this camera device")
    args = parser.parse_args()

    pc = RTCPeerConnection()

    async with ClientSession() as session:
        try:
            await run(pc, session, args.server, args.camera, args.device_id)
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # Ensure cleanup runs before exit
            await cleanup(args.device_id, args.server)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Publisher] Interrupted by user.")
