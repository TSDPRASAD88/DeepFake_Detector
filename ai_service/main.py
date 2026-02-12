from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import cv2
import numpy as np
from mtcnn import MTCNN
from torchvision import models, transforms
from PIL import Image
import shutil
import os

app = FastAPI()

# Optimized for Mac M1/M2/M3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Define the AI Model Architecture
class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super(DeepfakeClassifier, self).__init__()
        # Using efficientnet_b0 as the backbone
        self.network = models.efficientnet_b0(weights=None)
        
        # FIX: Change output features to 2 to match the downloaded weights
        num_ftrs = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(num_ftrs, 2)
        
        # Use Softmax for 2-class probability
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.network(x))

# 2. Initialize and Load the Model Weights
model = DeepfakeClassifier().to(device)
weights_path = 'weights/model.pth'

if os.path.exists(weights_path):
    try:
        # Load the raw weights
        state_dict = torch.load(weights_path, map_location=device)
        
        # FIX: Repair key mismatch
        new_state_dict = {}
        for k, v in state_dict.items():
            name = f"network.{k}" if not k.startswith("network.") else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        print(f"✅ Weights Loaded Successfully (2-class mode)")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
else:
    print("Warning: weights/model.pth not found.")

model.eval()

# 3. Initialize Face Detector
detector = MTCNN()

# 4. Define Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        # Save uploaded video
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}

        # Grab a frame from the middle
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"error": "Could not read video frame"}

        # 5. Detect face with safety
        faces = detector.detect_faces(frame)
        if not faces:
            return {"is_fake": False, "confidence": 0, "label": "REAL", "message": "No face detected"}

        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size == 0:
            return {"error": "Empty face crop"}

        # 6. Prediction Logic for 2-class output
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor) 
            # output is [prob_real, prob_fake]
            probabilities = output[0]
            real_prob = probabilities[0].item()
            fake_prob = probabilities[1].item()
        
        # Label is FAKE if fake_prob is higher than real_prob
        is_fake = fake_prob > real_prob
        confidence = fake_prob if is_fake else real_prob

        return {
            "is_fake": bool(is_fake),
            "confidence": round(float(confidence) * 100, 2),
            "label": "FAKE" if is_fake else "REAL"
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)