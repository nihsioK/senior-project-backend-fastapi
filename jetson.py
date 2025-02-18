import argparse
import asyncio
from aiohttp import ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
import cv2
from av import VideoFrame
from fractions import Fraction


class VideoCaptureTrack(VideoStreamTrack):
    """
    Class that captures video frames from a specified device and streams them.

    This class extends the VideoStreamTrack to provide functionality for capturing
    video frames from a camera device and streaming them as a video source. It can
    start and stop capturing frames upon command, providing frames asynchronously.
    The class makes use of OpenCV for video capture and frame processing.

    Attributes:
        cap (cv2.VideoCapture): The video capture object for accessing the camera device.
        fps (int): Frames per second for the stream.
        time_base (Fraction): Time base for calculating video frame timing.
        next_pts (int): Presentation timestamp of the next video frame.
        running (bool): A flag indicating whether the capturing is active or not.
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
    """
        Registers a camera with a server.

        This function sends an asynchronous HTTP POST request to a server endpoint
        to register a camera using the provided device ID. The server URL and
        camera information are included in the request.

        Parameters:
            device_id (str): The unique identifier for the camera being registered.
            server_url (str): The base URL of the server which processes the camera
                registration.

        Returns:
            bool: True if the camera is registered successfully (response status
                is 200 or 201), False otherwise.
    """
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
    """
        Asynchronous function to set the connection state of a device.

        This function sends a POST request to a specified server URL to update the
        connection state of a camera device identified by its `device_id`. The server
        endpoint takes the camera's connection state in its JSON payload. The function
        determines the success or failure of the request based on the HTTP response
        status codes.

        Args:
            device_id (str): Unique identifier of the camera device whose connection
                state is being updated.
            server_url (str): URL of the server to which the connection update request
                will be sent.
            connection (bool): Boolean value indicating whether the camera is connected
                (`True`) or disconnected (`False`).

        Returns:
            bool: `True` if the connection state was successfully updated
            (HTTP status code 200 or 201), otherwise `False`.

        Raises:
            Exception: Captures and handles all exceptions during the
            request execution phase, ensuring that the process does not
            disrupt other operations.
    """
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
    """
    Asynchronously controls the operation of a video stream based on commands received
    from a remote server. Continuously polls the server for commands to start or stop
    the video stream associated with a specific device.

    Parameters:
    device_id (str): Identifier for the target device whose video stream is to be controlled.
    video_track: A video track instance that supports start_stream and stop_stream methods
    to manage the video streaming.
    server_url (str): Base URL of the remote server to be polled for stream control commands.

    Raises:
    Exception: If any error occurs during the server request or while handling commands.

    Returns:
    None
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
                print(f"[Publisher] Error polling device_command: {e}")
            await asyncio.sleep(1)


async def run(pc, session, cloud_server_url, camera_device, device_id):
    # Configure ICE servers
    pc.configuration = RTCConfiguration(
        iceServers=[
            RTCIceServer(urls="stun:stun.relay.metered.ca:80"),
            RTCIceServer(
                urls="turn:global.relay.metered.ca:80",
                username="76d9bd49690a5fdc1e4e3760",
                credential="00pjOIhDISNLEWhB",
            ),
            RTCIceServer(
                urls="turn:global.relay.metered.ca:443",
                username="76d9bd49690a5fdc1e4e3760",
                credential="00pjOIhDISNLEWhB",
            ),
            RTCIceServer(
                urls="turns:global.relay.metered.ca:443",
                username="76d9bd49690a5fdc1e4e3760",
                credential="00pjOIhDISNLEWhB",
            ),
        ]
    )


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
            print(f"[Publisher] New ICE candidate: {candidate}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        print(f"[Publisher] ICE Connection State: {pc.iceConnectionState}")
        if pc.iceConnectionState in ["failed", "disconnected"]:
            print("[Publisher] ICE Connection failed. Attempting reconnection...")
            # Implement reconnection logic here

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

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
    """
    Summary:
    Cleans up resources or performs necessary operations before exiting. This
    includes disconnecting a device from the cloud server by setting its connection
    status to `False`.

    Args:
        device_id (str): The unique identifier of the device.
        cloud_server_url (str): The URL of the cloud server used for connectivity.

    Raises:
        Exception: If the cleanup process fails due to any error.
    """
    print("[Publisher] Cleaning up before exit...")
    try:
        await set_connection(device_id, cloud_server_url, False)
    except Exception as e:
        print(f"[Publisher] Cleanup failed: {e}")


async def main():
    """
    Initialize and execute the WebRTC sending process.

    This function is responsible for parsing command line arguments, establishing
    the RTCPeerConnection object, managing the WebRTC signaling and streaming
    logic, and ensuring proper cleanup upon termination or exception handling. The
    main process continuously sleeps to keep the coroutine running and listens for
    interrupt signals for shutdown operations.

    Args:
        None

    Returns:
        None

    Raises:
        asyncio.CancelledError: Raised when the asyncio task is explicitly cancelled.

    """
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
