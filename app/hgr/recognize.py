import cv2
import mediapipe as mp
import torch
import joblib
import asyncio

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load trained Transformer model
checkpoint = joblib.load('app/hgr/gesture_transformer.pkl')
model_state_dict = checkpoint['model_state_dict']
label_encoder = checkpoint['label_encoder']

# Define Transformer Model
class GestureTransformer(torch.nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128):
        super(GestureTransformer, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x.unsqueeze(1))
        x = x.mean(dim=1)
        return self.fc(x)

# Initialize model
input_dim = 63
num_classes = len(label_encoder.classes_)
num_heads = 4
if input_dim % num_heads != 0:
    num_heads = 1

model = GestureTransformer(input_dim=input_dim, num_classes=num_classes, num_heads=num_heads)
model.load_state_dict(model_state_dict)
model.to(device)  # Move model to GPU
model.eval()

# Convert model to TorchScript for faster inference
scripted_model = torch.jit.script(model)
scripted_model.to(device)  # Ensure the scripted model is also on GPU

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

async def recognize_gesture_async(frame):
    """
    Process a video frame asynchronously, extract hand landmarks, and predict gesture.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_data = [coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]

            # Convert to tensor and move to GPU
            landmark_tensor = torch.tensor(landmark_data, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = scripted_model(landmark_tensor)  # Use scripted model
                predicted_class = torch.argmax(output, dim=1).item()
                gesture_text = label_encoder.inverse_transform([predicted_class])[0]

            return gesture_text  # Return first detected gesture

    return None  # No hand detected
