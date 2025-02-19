from fastapi import APIRouter, HTTPException, Request
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from app.dependencies import publishers, subscriber_pcs
import re
from app.hgr.recognize import recognize_gesture

ice_servers = [
    RTCIceServer(urls="turn:senior-backend.xyz:3478", username="testuser", credential="supersecretpassword"),
    RTCIceServer(urls="turns:senior-backend.xyz:5349", username="testuser", credential="supersecretpassword")
]

router = APIRouter()

relay = MediaRelay()

class VideoProcessor(MediaStreamTrack):
    """
    Custom video processing track that extracts frames and detects gestures.
    """
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Recognize gesture
        gesture = recognize_gesture(img)
        if gesture:
            print(f"Detected Gesture: {gesture}")  # You can send this data to a frontend if needed

        return frame  # Return unmodified frame for WebRTC streaming

def remove_rtx(sdp: str) -> str:
    # Remove RTX-specific fmtp and rtcp-fb lines from the SDP
    sdp = re.sub(r'a=fmtp:\d+ apt=\d+\r\n', '', sdp)
    sdp = re.sub(r'a=rtcp-fb:\d+ nack\r\n', '', sdp)
    return sdp

@router.post("/offer")
async def offer(request: Request):
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

            @pc.on("iceconnectionstatechange")
            async def on_ice_state_change():
                print(f"[Cloud Server] Publisher '{device_id}' ICE state:", pc.iceConnectionState)
                if pc.iceConnectionState in ["failed", "disconnected"]:
                    print(f"[Cloud Server] Publisher '{device_id}' connection failed.")

            @pc.on("track")
            async def on_track(track):
                if track.kind == "video":
                    print(f"[Cloud Server] Publisher '{device_id}' video track received!")
                    publishers[device_id]["track"] = VideoProcessor(track)

            @pc.on("icecandidate")
            def on_ice_candidate(candidate):
                if candidate:
                    print(f"[Cloud Server] New ICE Candidate: {candidate}")
                else:
                    print("[Cloud Server] ICE Candidate gathering finished.")

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

            @pc.on("iceconnectionstatechange")
            async def on_ice_state_change():
                print(f"[Cloud Server] Subscriber (device {device_id}) ICE state:", pc.iceConnectionState)
                if pc.iceConnectionState in ["failed", "disconnected"]:
                    print(f"[Cloud Server] Subscriber connection failed for device {device_id}")

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