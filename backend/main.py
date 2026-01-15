import os
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
import librosa
from torchvision.models.video import swin3d_t
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ================== FASTAPI ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== MODEL ==================
class MFCC_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = MFCC_CNN().to(DEVICE)
model.load_state_dict(
    torch.load("mfcc_audio_detector.pth", map_location=DEVICE)
)
model.eval()
audio_model = model

# ================== AUDIO UTILS ==================
def extract_audio_from_video(video_path, out_wav):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        out_wav
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return out_wav


def extract_mfcc(audio_path, sr=16000, n_mfcc=40, max_len=300):
    audio, _ = librosa.load(audio_path, sr=sr)

    if len(audio) < sr:
        raise ValueError("Audio too short")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # Pad / truncate (EXACT AS TRAINING)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, max_len - mfcc.shape[1]))
        )
    else:
        mfcc = mfcc[:, :max_len]

    # Normalize (EXACT AS TRAINING)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    return mfcc

video_model = swin3d_t(weights=None)
video_model.head = torch.nn.Linear(video_model.head.in_features, 2)

video_model.load_state_dict(
    torch.load("best_swin_model.pth", map_location=DEVICE)
)

video_model.to(DEVICE)
video_model.eval()

def load_video_for_swin(path, max_frames=32):
    cap = cv2.VideoCapture(path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 30

    indices = np.linspace(0, total - 1, max_frames).astype(int)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    while len(frames) < max_frames:
        frames.append(frames[-1])

    frames = np.array(frames)

    # Center crop
    i = (256 - 224) // 2
    frames = frames[:, i:i+224, i:i+224, :]

    video = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0

    mean = torch.tensor([0.432, 0.394, 0.376]).view(3,1,1,1)
    std = torch.tensor([0.228, 0.221, 0.216]).view(3,1,1,1)

    video = (video - mean) / std
    return video.unsqueeze(0)

# ================== API ==================
@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    try:
        # ================= AUDIO MODEL =================
        # 1. Extract audio
        extract_audio_from_video(video_path, temp_audio)

        # 2. Extract MFCC
        mfcc = extract_mfcc(temp_audio)

        # 3. Prepare tensor (1, 1, 40, 300)
        audio_x = (
            torch.tensor(mfcc)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        # 4. Audio prediction
        with torch.no_grad():
            audio_probs = torch.softmax(audio_model(audio_x), dim=1)

        audio_real = float(audio_probs[0][0])
        audio_ai = float(audio_probs[0][1])

        # ================= VIDEO MODEL (SWIN3D) =================
        video_x = load_video_for_swin(video_path).to(DEVICE)

        with torch.no_grad():
            video_probs = torch.softmax(video_model(video_x), dim=1)

        video_real = float(video_probs[0][0])
        video_ai = float(video_probs[0][1])

        # ================= FUSION =================
        # Weighted fusion (audio slightly higher)
        final_ai = 0.6 * audio_ai + 0.4 * video_ai
        final_real = 1.0 - final_ai

        prediction = "AI-GENERATED" if final_ai >= 0.5 else "REAL"
        confidence = round(max(final_ai, final_real) * 100, 2)

        return {
            "prediction": prediction,
            "confidence": confidence,

            # Audio model output
            "audio_model": {
                "real_probability": round(audio_real, 4),
                "ai_probability": round(audio_ai, 4)
            },

            # Video model output
            "video_model": {
                "real_probability": round(video_real, 4),
                "ai_probability": round(video_ai, 4)
            },

            # Final fusion output
            "final_ai_probability": round(final_ai, 4),
            "final_real_probability": round(final_real, 4)
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


def predict_video_visual(video_path):
    x = load_video_for_swin(video_path).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(video_model(x), dim=1)

    return {
        "real_probability": float(probs[0][0]),
        "ai_probability": float(probs[0][1])
    }

