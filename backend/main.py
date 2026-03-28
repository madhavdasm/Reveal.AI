# ==============================
# FINAL FASTAPI BACKEND (NO-AUDIO SAFE VERSION)
# ==============================

import os
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
import librosa
import cv2

from torchvision.models.video import swin3d_t
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ================== APP ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== DEVICE ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_AI = 0
LABEL_REAL = 1


# ================== AUDIO MODEL ==================
class MFCC_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))

        x = nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


audio_model = MFCC_CNN().to(DEVICE)
audio_model.load_state_dict(torch.load("mfcc_cnn_model.pth", map_location=DEVICE))
audio_model.eval()


# ================== VIDEO MODEL ==================
class VideoSwinModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = swin3d_t(weights=None)
        in_features = self.backbone.head.in_features

        self.backbone.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)


video_model = VideoSwinModel().to(DEVICE)

checkpoint = torch.load("7k_swin_model.pth", map_location=DEVICE)

if isinstance(checkpoint, dict):
    state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
else:
    state_dict = checkpoint

print("Loading video model...")
video_model.load_state_dict(state_dict, strict=False)
video_model.eval()

VIDEO_THRESHOLD = checkpoint.get("threshold", 0.5) if isinstance(checkpoint, dict) else 0.5


# ================== VIDEO LOADER ==================
def load_video(path, max_frames=16):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 30

    indices = np.linspace(0, total - 1, max_frames).astype(int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    while len(frames) < max_frames:
        frames.append(frames[-1])

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    tensor_frames = []
    for f in frames:
        t = torch.tensor(f, dtype=torch.float32) / 255.0
        t = (t - mean) / std
        t = t.permute(2, 0, 1)
        tensor_frames.append(t)

    return torch.stack(tensor_frames).unsqueeze(0).float()


# ================== AUDIO ==================
def extract_audio(video_path, out_path):
    subprocess.run([
        "ffmpeg",
        "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        out_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_mfcc(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc


# ================== API ==================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    try:
        # ================= AUDIO =================
        audio_ai = None
        audio_real = None

        try:
            extract_audio(video_path, audio_path)

            # Check if audio exists
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                mfcc = extract_mfcc(audio_path)

                audio_x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

                with torch.no_grad():
                    a_probs = torch.softmax(audio_model(audio_x), dim=1)

                audio_ai = float(a_probs[0][1])
                audio_real = float(a_probs[0][0])
            else:
                print("⚠️ No audio detected")

        except Exception as e:
            print("⚠️ Audio processing failed:", e)

        # ================= VIDEO =================
        video_x = load_video(video_path).to(DEVICE)

        with torch.no_grad():
            v_probs = torch.softmax(video_model(video_x), dim=1)

        video_ai = float(v_probs[0][LABEL_AI])
        video_real = float(v_probs[0][LABEL_REAL])

        # ================= FUSION =================
        if audio_ai is not None:
            final_ai = 0.6 * audio_ai + 0.4 * video_ai
        else:
            final_ai = video_ai  # fallback if no audio

        final_real = 1 - final_ai

        prediction = "AI-GENERATED" if final_ai >= 0.5 else "REAL"

        return {
            "prediction": prediction,
            "confidence": round(max(final_ai, final_real) * 100, 2),

            "audio_model": {
                "real_probability": round(audio_real, 4) if audio_real is not None else None,
                "ai_probability": round(audio_ai, 4) if audio_ai is not None else None,
            },

            "video_model": {
                "real_probability": round(video_real, 4),
                "ai_probability": round(video_ai, 4),
                "threshold_used": round(VIDEO_THRESHOLD, 4),
            },

            "final_ai_probability": round(final_ai, 4),
            "final_real_probability": round(final_real, 4),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)


# ================== RUN ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)