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

# Optional: enable debug logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()
relay = MediaRelay()

# Define ICE servers (TURN)
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


@router.post("/offer")
async def offer(request: Request):
    """
    Handle incoming offers from both publisher and subscriber roles.
    """
    try:
        params = await request.json()
        role = params.get("role")
        device_id = params.get("device_id")
        sdp = params.get("sdp")
        type_ = params.get("type")

        # ------------------------------------------------
        # PUBLISHER
        # ------------------------------------------------
        if role == "publisher":
            if not device_id:
                raise HTTPException(status_code=400, detail="Missing device_id for publisher")

            pc = RTCPeerConnection(RTC_CONFIG)
            publishers[device_id] = {"pc": pc, "track": None, "streaming": False}

            @pc.on("track")
            async def on_track(track):
                """
                Handle incoming tracks from the publisher.
                We also ignore the 'video/rtx' track to avoid
                'No decoder found for MIME type video/rtx' errors.
                """
                if track.kind == "video":
                    try:
                        # Some aiortc versions do not have track.codec
                        # so we use try/except to avoid AttributeError.
                        mime_type = track.codec.mimeType
                        if mime_type == "video/rtx":
                            logging.info("[Cloud Server] Received video/rtx track, stopping it.")
                            track.stop()
                            return
                    except AttributeError:
                        pass

                    logging.info(f"[Cloud Server] Publisher '{device_id}' main video track received!")
                    publishers[device_id]["track"] = track

            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate:
                    logging.debug(f"[Cloud Server] Publisher ICE candidate: {candidate.to_sdp()}")
                else:
                    logging.info("[Cloud Server] Publisher ICE gathering complete.")

            # Set remote SDP from the publisher
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
            # Create and set local SDP answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }

        # ------------------------------------------------
        # SUBSCRIBER
        # ------------------------------------------------
        elif role == "subscriber":
            if (
                not device_id or
                device_id not in publishers or
                publishers[device_id]["track"] is None
            ):
                return {"error": f"No active publisher for device {device_id}"}

            pc = RTCPeerConnection(RTC_CONFIG)
            # Relay the publisher's video track to the subscriber
            local_video = relay.subscribe(publishers[device_id]["track"])
            pc.addTrack(local_video)

            @pc.on("iceconnectionstatechange")
            async def on_ice_state_change():
                logging.info(
                    f"[Cloud Server] Subscriber (device {device_id}) ICE state: {pc.iceConnectionState}"
                )

            @pc.on("icecandidate")
            async def on_ice_candidate(candidate):
                if candidate:
                    logging.debug(f"[Cloud Server] Subscriber ICE candidate: {candidate.to_sdp()}")
                else:
                    logging.info("[Cloud Server] Subscriber ICE gathering complete.")

            # Set remote SDP from the subscriber
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
            # Create and set local SDP answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            subscriber_pcs.add(pc)
            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }

        else:
            return {"error": "Invalid role specified"}

    except Exception as e:
        logging.error("[Cloud Server] Exception in /offer: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_stream")
async def set_stream(request: Request):
    """
    Sets the "streaming" flag for a given publisher device,
    which can be polled by the Jetson client.
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
