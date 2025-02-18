from fastapi import APIRouter, HTTPException, Request
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from aiortc.contrib.media import MediaRelay

from app.dependencies import publishers, subscriber_pcs

router = APIRouter()

relay = MediaRelay()

ICE_CONFIGURATION = RTCConfiguration(
    iceServers=[
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:YOUR_VPS_IP:3478"],
            "username": "turnuser",
            "credential": "turnpass"
        }
    ]
)


@router.post("/offer")
async def offer(request: Request):
    try:
        params = await request.json()
        role = params.get("role")
        device_id = params.get("device_id")
        sdp = params.get("sdp")
        type_ = params.get("type")

        if role == "publisher":
            if not device_id:
                raise HTTPException(status_code=400, detail="Missing device_id for publisher")

            pc = RTCPeerConnection(configuration=ICE_CONFIGURATION)
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
                    publishers[device_id]["track"] = track

            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

        elif role == "subscriber":
            if not device_id or device_id not in publishers or publishers[device_id]["track"] is None:
                return {"error": f"No active publisher for device {device_id}"}

            pc = RTCPeerConnection(configuration=ICE_CONFIGURATION)
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