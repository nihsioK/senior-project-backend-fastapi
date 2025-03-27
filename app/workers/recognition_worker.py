import redis
import base64
import cv2
import numpy as np
import json
import logging
from collections import defaultdict, deque
from app.services.gesture_statistics_service import ActionStatisticService
from app.repositories.notification_repository import NotificationRepository
from app.models.notification_models import NotificationType
from app.dependencies import get_db

# ----- Gesture Recognition Imports and Global Setup -----
import mediapipe as mp
import torch
import torch.nn.functional as F
from .model import GestureGCN

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define class labels for gesture recognition
label_names = {0: "thumbs_down", 1: "rock", 2: "thumbs_up", 3: "peace", 4: "palm", 5: "fist"}

# Initialize the gesture recognition model and set it to evaluation mode.
try:
    model = GestureGCN(num_classes=6)
    model.load_state_dict(torch.load("trained_gesture_gcn.pth"))
    model.eval()
except Exception as e:
    logger.error(f"Failed to load gesture model: {e}")
    raise

# Initialize mediapipe hands detector (using only one hand for simplicity)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)


def softmax_with_temperature(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=1)


# Initialize services
action_service = ActionStatisticService()
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)

# Store frames per device (each device gets a queue of max 10 frames)
frame_queues = defaultdict(lambda: deque(maxlen=10))


def predict_from_frame(frame):
    """
    Process a single frame to perform gesture recognition.
    Returns a tuple (prediction, confidence).
    If no hand is detected or if the prediction is uncertain, returns ("I'm not sure", 0.0).
    """
    try:
        # Convert frame to RGB as mediapipe expects RGB images.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None, 0.0

        # For simplicity, process the first detected hand.
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract keypoints for x, y, z in order.
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.append(lm.x)
        for lm in hand_landmarks.landmark:
            keypoints.append(lm.y)
        for lm in hand_landmarks.landmark:
            keypoints.append(lm.z)

        # Prepare input tensor: reshape to (1, 3, 1, 21) as expected by the model.
        keypoints = np.array(keypoints).reshape(3, 21)
        keypoints = np.expand_dims(keypoints, axis=1)  # (3, 1, 21)
        keypoints = np.expand_dims(keypoints, axis=0)  # (1, 3, 1, 21)
        input_tensor = torch.tensor(keypoints, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            # Apply softmax with temperature scaling (temperature=2.0 can be adjusted)
            probs = softmax_with_temperature(output, temperature=2.0)
            top2_probs, top2_classes = torch.topk(probs, 2, dim=1)
            confidence = top2_probs[0][0].item()
            second_confidence = top2_probs[0][1].item()

            # If the difference between the top two probabilities is small, consider the prediction uncertain.
            if confidence - second_confidence < 0.95:
                return "I'm not sure", 0.0
            else:
                prediction = label_names[top2_classes[0][0].item()]
                return prediction, confidence
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "I'm not sure", 0.0


def process_frames(frames):
    """
    Process accumulated frames using the gesture recognition model.
    For each frame, if a hand is detected and the model is confident, record the prediction.
    The final action is chosen as the prediction with the highest confidence.
    """
    predictions = []
    for frame in frames:
        pred, conf = predict_from_frame(frame)
        if pred and pred != "I'm not sure":
            predictions.append((pred, conf))

    if not predictions:
        return "I'm not sure"

    # Choose the prediction with the highest confidence.
    best_prediction = max(predictions, key=lambda x: x[1])[0]
    return best_prediction


def recognition_worker(db_session, notification_repository: NotificationRepository):
    """
    Continuously processes frames from a Redis Stream.
    After accumulating 10 frames for a given device, it runs gesture recognition,
    sends the result via Redis Pub/Sub, and (if needed) issues an alert.
    """
    try:
        while True:
            try:
                messages = redis_client.xread({"video_frames": "$"}, count=1, block=500)  # Blocking read
                logger.info(f"Received {len(messages)} frames")

                for stream, message_list in messages:
                    for message_id, data in message_list:
                        device_id = data[b"device_id"].decode("utf-8")
                        frame_encoded = data[b"frame"].decode("utf-8")

                        # Decode frame from base64 to image
                        np_arr = np.frombuffer(base64.b64decode(frame_encoded), np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        frame_queues[device_id].append(img)

                        # Once 10 frames have been accumulated, process them
                        if len(frame_queues[device_id]) >= 1:
                            action_result = process_frames(frame_queues[device_id])
                            logger.info(f"{device_id}: {action_result}")
                            frame_queues[device_id].clear()  # Clear the queue after processing

                            # Send the recognition result back via Redis Pub/Sub
                            redis_client.publish(
                                "action_results", json.dumps({"device_id": device_id, "action": action_result})
                            )

                            # Example: if the recognized action is "rock", send an alert and create a notification.
                            if action_result == "rock":
                                alert_message = {
                                    "device_id": device_id,
                                    "message": f"On {device_id} detected unusual activity: {action_result}",
                                }
                                redis_client.publish("alerts", json.dumps(alert_message))
                                notification_repository.create_manual(
                                    notification_type=NotificationType.CRITICAL,
                                    message=alert_message["message"],
                                    camera_id=alert_message["device_id"],
                                )

                            try:
                                action_service.process_action(None, device_id, action_result)
                            except Exception as e:
                                logger.error(f"Error processing action: {e}")

            except redis.exceptions.RedisError as re:
                logger.error(f"Redis error: {re}")
                # Add a small delay to prevent rapid reconnection attempts
                import time

                time.sleep(2)

    except Exception as e:
        logger.error(f"Unexpected error in recognition worker: {e}")
    finally:
        db_session.close()


if __name__ == "__main__":
    logger.info("Starting recognition worker")
    db_session = next(get_db())
    notification_repository = NotificationRepository(db_session)
    try:
        recognition_worker(db_session, notification_repository)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    finally:
        db_session.close()
