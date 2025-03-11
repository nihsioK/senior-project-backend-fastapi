from fastapi import APIRouter, HTTPException, Request
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
import asyncio
import re
from app.dependencies import publishers, subscriber_pcs
import redis
import cv2
import base64

ice_servers = [
    RTCIceServer(urls="turn:46.8.31.7:3478", username="testuser", credential="supersecretpassword"),
    RTCIceServer(urls="turns:46.8.31.7:5349", username="testuser", credential="supersecretpassword")
]

router = APIRouter()
relay = MediaRelay()

frame_queue = asyncio.Queue()

redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)

async def frame_processing_worker():
    """
    Background worker that processes frames from the queue.
    Tracks time taken for the first detected gesture.
    """

    while True:
        device_id, frame = await frame_queue.get()
        if frame is None:
            break

async def extract_frames(device_id, track):
    """
    Extracts frames from the WebRTC track and queues them for processing.
    Tracks the time taken to receive 30 frames.
    """
    counter = 0
    while True:
        try:

            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")

            _, buffer = cv2.imencode('.jpg', img)
            encoded_frame = base64.b64encode(buffer).decode("utf-8")

            redis_client.xadd("video_frames", {"device_id": device_id, "frame": encoded_frame})
            if counter % 10 == 0:
                print("Processed {} frames".format(counter))
            counter += 1
            await frame_queue.put((device_id, img))
        except Exception as e:
            print(f"[Frame Extraction] Error processing video from '{device_id}':", e)
            break  # Stop processing if track is closed


def remove_rtx(sdp: str) -> str:
    """
    Cleans up the SDP to remove unwanted RTX-related attributes.
    """
    sdp = re.sub(r'a=fmtp:\d+ apt=\d+\r\n', '', sdp)
    sdp = re.sub(r'a=rtcp-fb:\d+ nack\r\n', '', sdp)
    return sdp

@router.post("/offer")
async def offer(request: Request):
    """
    Handles WebRTC offer, assigns a track for the publisher, and sets up the connection.
    """
    try:
        params = await request.json()
        role = params.get("role")
        device_id = params.get("device_id")
        sdp = params.get("sdp")
        type_ = params.get("type")

        sdp = remove_rtx(sdp)

        if role == "publisher":
            if not device_id:
                raise HTTPException(status_code=400, detail="Missing device_id for publisher")

            pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
            publishers[device_id] = {"pc": pc, "track": None, "streaming": False}

            @pc.on("track")
            async def on_track(track):
                if track.kind == "video":
                    print(f"[Cloud Server] Publisher '{device_id}' video track received!")
                    publishers[device_id]["track"] = track  # Store the original track
                    asyncio.create_task(extract_frames(device_id, track))  # Start async extraction

            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

        elif role == "subscriber":
            if not device_id or device_id not in publishers or publishers[device_id]["track"] is None:
                return {"error": f"No active publisher for device {device_id}"}

            pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
            local_video = relay.subscribe(publishers[device_id]["track"])
            pc.addTrack(local_video)

            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            subscriber_pcs.add(pc)

            return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

        else:
            return {"error": "Invalid role specified"}
    except Exception as e:
        print("[Cloud Server] Exception in /offer:", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set_stream")
async def set_stream(request: Request):
    """
    Enables or disables streaming for a publisher.
    """
    try:
        params = await request.json()
        device_id = params.get("device_id")
        stream_flag = params.get("stream")

        if not device_id:
            raise HTTPException(status_code=400, detail="Missing device_id")
        if device_id not in publishers:
            raise HTTPException(status_code=404, detail=f"Device {device_id} not found")

        publishers[device_id]["streaming"] = bool(stream_flag)
        print(f"[Cloud Server] Device '{device_id}' streaming set to {stream_flag}")

        return {"device_id": device_id, "streaming": publishers[device_id]["streaming"]}
    except Exception as e:
        print("[Cloud Server] Exception in /set_stream:", e)
        raise HTTPException(status_code=500, detail=str(e))
