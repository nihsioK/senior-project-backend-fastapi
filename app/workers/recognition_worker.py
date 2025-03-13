import redis
import base64
import cv2
import numpy as np
import json
import random
from collections import defaultdict, deque
from app.services.gesture_statistics_service import ActionStatisticService

action_statistics_service = ActionStatisticService()

redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)

# Store frames per device
frame_queues = defaultdict(lambda: deque(maxlen=10))  # Each device gets a queue of max 10 frames


def process_frames(frames):
    """
    Process accumulated frames and determine the most likely action.
    """
    # Simulate action recognition (replace with your model inference logic)
    actions = ["thumbs up", "thumbs down", "peace", "rock"]
    chosen_action = random.choice(actions)

    return chosen_action


def recognition_worker():
    """
    Continuously processes frames from Redis Stream and sends results after accumulating 10 frames.
    """
    while True:
        messages = redis_client.xread({"video_frames": "$"}, count=1, block=500)  # Blocking read
        print("Received {} frames".format(len(messages)))
        for stream, message_list in messages:
            for message_id, data in message_list:
                device_id = data[b"device_id"].decode("utf-8")
                frame = data[b"frame"].decode("utf-8")

                # Decode frame and add it to the queue
                np_arr = np.frombuffer(base64.b64decode(frame), np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                frame_queues[device_id].append(img)

                # If we have 10 frames, process and send result
                if len(frame_queues[device_id]) == 10:
                    action_result = process_frames(list(frame_queues[device_id]))  # Process all 10 frames
                    print(f"{device_id}: {action_result}")
                    frame_queues[device_id].clear()  # Clear queue after processing

                    # Send recognition result back via Redis Pub/Sub
                    redis_client.publish(
                        "action_results", json.dumps({"device_id": device_id, "action": action_result})
                    )

                    if action_result == "rock":
                        alert_message = {
                            "device_id": device_id,
                            "message": f"On {device_id} detected unusual activity: {action_result}",
                        }
                        redis_client.publish("alerts", json.dumps(alert_message))

                    action_statistics_service.process_action(None, device_id, action_result)


if __name__ == "__main__":
    print("Starting recognition worker")
    recognition_worker()
