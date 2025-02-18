from aiortc import RTCPeerConnection, RTCIceServer, RTCConfiguration
import asyncio

async def test_turn():
    config = RTCConfiguration(iceServers=[
        RTCIceServer(urls="turn:senior-backend.xyz:3478", username="testuser", credential="supersecretpassword"),
        RTCIceServer(urls="turns:senior-backend.xyz:5349", username="testuser", credential="supersecretpassword")
    ])

    pc = RTCPeerConnection(configuration=config)

    # Создаём DataChannel
    channel = pc.createDataChannel("testChannel")

    @pc.on("icecandidate")
    def on_ice_candidate(event):
        if event.candidate:
            print(f"New ICE Candidate: {event.candidate}")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Ожидаем ICE-кандидатов
    await asyncio.sleep(5)

    # Получаем ICE-кандидаты корректно
    ice_candidates = []
    for transceiver in pc.getTransceivers():
        if transceiver.sender.transport:
            candidate = transceiver.sender.transport.iceGatherer.getLocalCandidates()
            ice_candidates.extend(candidate)

    print("✅ ICE candidates:", ice_candidates)

asyncio.run(test_turn())
