import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
from model import GestureGCN  # Make sure model.py is in the same directory

# Map numeric labels to gesture names
label_names = {0: "thumbs_down", 1: "rock", 2: "thumbs_up", 3: "peace", 4: "palm", 5: "fist"}

# Initialize MediaPipe and model once globally
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Load gesture model
try:
    model = GestureGCN(num_classes=6)
    model.load_state_dict(torch.load("trained_gesture_gcn.pth", map_location=torch.device("cpu")))
    model.eval()
    print("[Recognition] Model loaded successfully.")
except Exception as e:
    print(f"[Recognition] Failed to load gesture model: {e}")
    raise


def softmax_with_temperature(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=1)


def predict_from_frame(frame):
    """
    Predicts gesture from a single frame.
    Returns (label, confidence) or ("I'm not sure", 0.0) if uncertain.
    """
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return "I'm not sure", 0.0

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
        print(f"[Recognition] Prediction error: {e}")
        return "I'm not sure", 0.0
