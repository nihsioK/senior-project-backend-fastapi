import cv2
import mediapipe as mp
import torch
import joblib

# Load trained Transformer model
checkpoint = joblib.load('gesture_transformer.pkl')
model_state_dict = checkpoint['model_state_dict']
label_encoder = checkpoint['label_encoder']


# Define Transformer Model
class GestureTransformer(torch.nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128):
        super(GestureTransformer, self).__init__()
        assert input_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x.unsqueeze(1))  # Add sequence dimension
        x = x.mean(dim=1)  # Aggregate across sequence
        return self.fc(x)


# Initialize model
input_dim = 63  # 21 landmarks * 3 (x, y, z)
num_classes = len(label_encoder.classes_)
num_heads = 4
if input_dim % num_heads != 0:
    num_heads = 1  # Ensure divisibility by setting num_heads to 1 if necessary

model = GestureTransformer(input_dim=input_dim, num_classes=num_classes, num_heads=num_heads)
model.load_state_dict(model_state_dict)
model.eval()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


def recognize_gesture(frame):
    """
    Process a video frame, extract hand landmarks, and predict gesture.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_data = []
            for landmark in hand_landmarks.landmark:
                landmark_data.extend([landmark.x, landmark.y, landmark.z])

            # Convert to tensor and predict
            landmark_tensor = torch.tensor(landmark_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = model(landmark_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                gesture_text = label_encoder.inverse_transform([predicted_class])[0]

            return gesture_text  # Return first detected gesture

    return None  # No hand detected
