import logging
from fastapi import APIRouter, HTTPException, Request
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer
)
from aiortc.contrib.media import MediaRelay
from app.dependencies import publishers, subscriber_pcs

# ----- BEGIN: Override RTX Decoder -----
from aiortc.codecs import get_decoder as original_get_decoder
import aiortc.codecs

class DummyDecoder:
    async def decode(self, packet):
        # Simply return no frames.
        return []

def get_decoder(codec):
    if codec.mimeType.lower() == "video/rtx":
        logging.info("Using dummy decoder for video/rtx")
        return DummyDecoder()
    return original_get_decoder(codec)

aiortc.codecs.get_decoder = get_decoder
# ----- END: Override RTX Decoder -----

logging.basicConfig(level=logging.INFO)
router = APIRouter()
relay = MediaRelay()

# For simplicity, use a public STUN server.
ICE_SERVERS = [
    RTCIceServer(
        urls=["stun:stun.l.google.com:19302"]
    )
]

RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)

def remove_rtx_from_sdp(sdp: str) -> str:
    """
    Optionally, remove any SDP lines containing "rtx".
    (You can comment this out if the dummy decoder handles RTX.)
    """
    lines = sdp.splitlines()
    filtered = [line for line in lines if "rtx" not in line.lower()]
    return "\r\n".join(filtered) + "\r\n"

@router.post("/offer")
async def offer(request: Request):
    """
    Accepts an SDP offer from a publisher or subscriber.
    For publishers, we remove RTX lines from the incoming SDP.
    """
    try:
        params = await request.json()
        role = params.get("role")
        device_id = params.get("device_id")
        sdp = params.get("sdp")
        type_ = params.get("type")

        if role == "publisher":
            if not device_id:
                raise HTTPException(status_code=400, detail="Missing device_id for publisher")
            # Optionally remove RTX lines:
            sdp = remove_rtx_from_sdp(sdp)

            pc = RTCPeerConnection(RTC_CONFIG)
            publishers[device_id] = {"pc": pc, "track": None, "streaming": True}

            @pc.on("track")
            async def on_track(track):
                if track.kind == "video":
                    logging.info(f"[Cloud Server] Publisher '{device_id}' main video track received!")
                    publishers[device_id]["track"] = track

            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate:
                    logging.debug(f"[Cloud Server] Publisher ICE candidate: {candidate.to_sdp()}")
                else:
                    logging.info("[Cloud Server] Publisher ICE gathering complete.")

            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

        elif role == "subscriber":
            if not device_id or device_id not in publishers or publishers[device_id]["track"] is None:
                return {"error": f"No active publisher for device {device_id}"}

            pc = RTCPeerConnection(RTC_CONFIG)
            local_video = relay.subscribe(publishers[device_id]["track"])
            pc.addTrack(local_video)

            @pc.on("iceconnectionstatechange")
            async def on_ice_state_change():
                logging.info(f"[Cloud Server] Subscriber (device {device_id}) ICE state: {pc.iceConnectionState}")

            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate:
                    logging.debug(f"[Cloud Server] Subscriber ICE candidate: {candidate.to_sdp()}")
                else:
                    logging.info("[Cloud Server] Subscriber ICE gathering complete.")

            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            subscriber_pcs.add(pc)
            return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

        else:
            return {"error": "Invalid role specified"}

    except Exception as e:
        logging.error("[Cloud Server] Exception in /offer: %s", e)
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
        logging.info(f"[Cloud Server] Device '{device_id}' streaming set to {stream_flag}")
        return {"device_id": device_id, "streaming": publishers[device_id]["streaming"]}
    except Exception as e:
        logging.error("[Cloud Server] Exception in /set_stream: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
