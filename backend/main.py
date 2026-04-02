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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="../"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("../index.html")

@app.get("/app.html")
async def serve_app_page():
    return FileResponse("../app.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_AI = 0
LABEL_REAL = 1


def f32(x):
    if hasattr(x, "float"):
        return float(x.float())
    return float(x)


# ================== AUDIO MODEL ==================
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


audio_model = MFCC_CNN().to(DEVICE)
audio_model.load_state_dict(
    torch.load("mfcc_audio_detector.pth", map_location=DEVICE)
)
audio_model.eval()


# ================== VIDEO MODEL ==================
class VideoSwinModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = swin3d_t(weights=None)
        in_features = self.backbone.head.in_features

        # FIX: matches checkpoint (head.1.weight)
        self.backbone.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)


video_model = VideoSwinModel().to(DEVICE)

checkpoint = torch.load("7k_swin_model.pth", map_location=DEVICE)

if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

print("Loading video model...")
video_model.load_state_dict(state_dict)
video_model.eval()

VIDEO_THRESHOLD = checkpoint.get("threshold", 0.5) if isinstance(checkpoint, dict) else 0.5


# ================== VIDEO LOADER (FIXED FLOAT32) ==================
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

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    tensor_frames = []
    for f in frames:
        t = torch.tensor(f, dtype=torch.float32) / 255.0
        t = (t - mean) / std
        t = t.permute(2, 0, 1)
        tensor_frames.append(t)

    video = torch.stack(tensor_frames).unsqueeze(0).float()
    return video


# ================== AUDIO ==================
def extract_audio(video_path, out_path):
    cmd = [
        r"C:\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe",
        "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
        # AUDIO
        extract_audio(video_path, audio_path)
        mfcc = extract_mfcc(audio_path)

        audio_x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            a_probs = torch.softmax(audio_model(audio_x), dim=1)

        audio_ai = f32(a_probs[0][1])
        audio_real = f32(a_probs[0][0])

        # VIDEO
        video_x = load_video(video_path).to(DEVICE).float()

        with torch.no_grad():
            v_probs = torch.softmax(video_model(video_x), dim=1)

        video_ai = f32(v_probs[0][LABEL_AI])
        video_real = f32(v_probs[0][LABEL_REAL])

        # FUSION
        final_ai = 0.6 * audio_ai + 0.4 * video_ai
        final_real = 1 - final_ai

        prediction = "AI-GENERATED" if final_ai >= 0.5 else "REAL"

        return {
            "prediction": prediction,
            "confidence": float(round(max(final_ai, final_real) * 100, 2)),

            "audio_model": {
                "real_probability": float(round(audio_real, 4)),
                "ai_probability": float(round(audio_ai, 4)),
            },

            "video_model": {
                "real_probability": float(round(video_real, 4)),
                "ai_probability": float(round(video_ai, 4)),
                "threshold_used": float(round(VIDEO_THRESHOLD, 4)),
            },

            "final_ai_probability": float(round(final_ai, 4)),
            "final_real_probability": float(round(final_real, 4)),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
