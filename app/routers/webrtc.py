from fastapi import APIRouter, HTTPException, Request
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
import requests
from app.dependencies import publishers, subscriber_pcs

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



router = APIRouter()

relay = MediaRelay()

def get_ice_configuration():
    return RTCConfiguration(iceServers=get_turn_credentials())


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
                    publishers[device_id]["track"] = track

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