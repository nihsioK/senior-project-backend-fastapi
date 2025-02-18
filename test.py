import requests
from aiortc import RTCConfiguration, RTCIceServer

# Metered.ca API Key
API_KEY = "2492bffdb915ca4e706d051ea6bb8de323ff"

def get_turn_credentials():
    url = f"https://senior.metered.live/api/v1/turn/credentials?apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        turn_servers = response.json()
        return [
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username", ""),
                credential=server.get("credential", "")
            )
            for server in turn_servers
        ]
    else:
        print(f"Failed to fetch TURN credentials: {response.status_code}")
        return []

# Fetch credentials dynamically before WebRTC connection
ICE_CONFIGURATION = RTCConfiguration(iceServers=get_turn_credentials())
