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

import mediapipe as mp
import torch
import torch.nn.functional as F
from .model import GestureGCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

label_names = {0: "thumbs_down", 1: "rock", 2: "thumbs_up", 3: "peace", 4: "palm", 5: "fist"}

try:
    model = GestureGCN(num_classes=6)
    model.load_state_dict(torch.load("trained_gesture_gcn.pth"))
    model.eval()
except Exception as e:
    logger.error(f"Failed to load gesture model: {e}")
    raise

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)


def softmax_with_temperature(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=1)


action_service = ActionStatisticService()
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)

frame_queues = defaultdict(lambda: deque(maxlen=10))


def predict_from_frame(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None, 0.0

        hand_landmarks = results.multi_hand_landmarks[0]

        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.append(lm.x)
        for lm in hand_landmarks.landmark:
            keypoints.append(lm.y)
        for lm in hand_landmarks.landmark:
            keypoints.append(lm.z)

        keypoints = np.array(keypoints).reshape(3, 21)
        keypoints = np.expand_dims(keypoints, axis=1)
        keypoints = np.expand_dims(keypoints, axis=0)
        input_tensor = torch.tensor(keypoints, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            probs = softmax_with_temperature(output, temperature=2.0)
            top2_probs, top2_classes = torch.topk(probs, 2, dim=1)
            confidence = top2_probs[0][0].item()
            second_confidence = top2_probs[0][1].item()

            if confidence - second_confidence < 0.8:
                return "I'm not sure", 0.0
            else:
                prediction = label_names[top2_classes[0][0].item()]
                return prediction, confidence
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "I'm not sure", 0.0


def process_frames(frames):
    predictions = []
    for frame in frames:
        pred, conf = predict_from_frame(frame)
        if pred and pred != "I'm not sure":
            predictions.append((pred, conf))

    if not predictions:
        return "I'm not sure"

    best_prediction = max(predictions, key=lambda x: x[1])[0]
    return best_prediction


def recognition_worker(db_session, notification_repository: NotificationRepository):
    try:
        while True:
            try:
                messages = redis_client.xread({"video_frames": "$"}, count=1, block=500)
                logger.info(f"Received {len(messages)} frames")

                for stream, message_list in messages:
                    for message_id, data in message_list:
                        device_id = data[b"device_id"].decode("utf-8")
                        frame_encoded = data[b"frame"].decode("utf-8")

                        np_arr = np.frombuffer(base64.b64decode(frame_encoded), np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        frame_queues[device_id].append(img)

                        if len(frame_queues[device_id]) >= 1:
                            action_result = process_frames(frame_queues[device_id])
                            logger.info(f"{device_id}: {action_result}")
                            frame_queues[device_id].clear()

                            redis_client.publish(
                                "action_results", json.dumps({"device_id": device_id, "action": action_result})
                            )

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
