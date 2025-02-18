from aiortc import RTCConfiguration, RTCIceServer
import requests


def get_turn_credentials():
    API_KEY = "2492bffdb915ca4e706d051ea6bb8de323ff"
    TURN_CREDENTIALS_URL = f"https://senior.metered.live/api/v1/turn/credentials?apiKey={API_KEY}"

    response = requests.get(TURN_CREDENTIALS_URL)
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


# Apply the dynamically fetched credentials
ICE_CONFIGURATION = RTCConfiguration(iceServers=get_turn_credentials())

print("Final ICE Configuration:", ICE_CONFIGURATION.iceServers)
