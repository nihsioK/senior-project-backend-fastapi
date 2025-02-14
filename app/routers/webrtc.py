import logging
from fastapi import APIRouter, HTTPException, Request
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay

from app.dependencies import publishers, subscriber_pcs

logging.basicConfig(level=logging.INFO)
router = APIRouter()
relay = MediaRelay()

# Define ICE servers (using your TURN server)
ICE_SERVERS = [
    RTCIceServer(
        urls=[
            "turn:46.8.31.7:3478?transport=udp",
            "turn:46.8.31.7:3478?transport=tcp"
        ],
        username="webrtcuser",
        credential="strongpassword",
        credentialType="password"
    )
]

RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)

def remove_rtx_from_sdp(sdp: str) -> str:
    """
    Remove any SDP lines containing 'rtx' (case-insensitive) so that
    no retransmission (RTX) payload is negotiated.
    """
    lines = sdp.splitlines()
    filtered = [line for line in lines if "rtx" not in line.lower()]
    return "\r\n".join(filtered) + "\r\n"

@router.post("/offer")
async def offer(request: Request):
    """
    Handle incoming SDP offers from publisher or subscriber.
    For publishers, remove RTX lines from the SDP.
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

            # Remove RTX-related lines from the SDP
            sdp = remove_rtx_from_sdp(sdp)

            pc = RTCPeerConnection(RTC_CONFIG)
            publishers[device_id] = {"pc": pc, "track": None, "streaming": False}

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

    except StopIteration as se:
        logging.error("[Cloud Server] StopIteration caught: %s", se)
        raise HTTPException(status_code=500, detail="Internal SDP negotiation error")
    except Exception as e:
        logging.error("[Cloud Server] Exception in /offer: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_stream")
async def set_stream(request: Request):
    """
    Sets the "streaming" flag for a given publisher device.
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
        logging.info(f"[Cloud Server] Device '{device_id}' streaming set to {stream_flag}")
        return {"device_id": device_id, "streaming": publishers[device_id]["streaming"]}
    except Exception as e:
        logging.error("[Cloud Server] Exception in /set_stream: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
