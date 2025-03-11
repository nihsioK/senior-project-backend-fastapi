import asyncio
import websockets

async def receive_actions(device_id):
    uri = f"wss://senior-backend.xyz/ws/actions/{device_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Connected to action results for device: {device_id}")

        while True:
            message = await websocket.recv()
            print("Received:", message)

if __name__ == "__main__":
    device_id = "camera54"  # Replace with actual device ID
    asyncio.run(receive_actions(device_id))
