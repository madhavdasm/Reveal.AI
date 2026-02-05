import os
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torchvision.models.video import swin3d_t
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import base64
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="../"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("../index.html")

# ================== FASTAPI ==================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== GRAD-CAM IMPLEMENTATION ==================

class GradCAM:
    """Generic Grad-CAM implementation"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.handlers = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def full_backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.handlers.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.handlers.append(
            self.target_layer.register_full_backward_hook(full_backward_hook)
        )
    
    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input to the model
            target_class: Target class index (None = predicted class)
        
        Returns:
            cam: Grad-CAM heatmap normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=tuple(range(2, len(gradients.shape))), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Robust normalization with percentile clipping for better contrast
        cam_min = np.percentile(cam, 5)
        cam_max = np.percentile(cam, 95)
        
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Clip to [0, 1]
        cam = np.clip(cam, 0, 1)
        
        return cam


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation for better localization"""
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Calculate alpha weights
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        # Avoid division by zero
        alpha = grad_2 / (2 * grad_2 + (grad_3 * activations).sum(dim=tuple(range(2, len(gradients.shape))), keepdim=True) + 1e-8)
        
        # ReLU on gradients
        relu_grad = F.relu(class_loss.exp() * gradients)
        
        # Weighted combination
        weights = (alpha * relu_grad).sum(dim=tuple(range(2, len(gradients.shape))), keepdim=True)
        
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Robust normalization with percentile clipping for better contrast
        cam_min = np.percentile(cam, 5)
        cam_max = np.percentile(cam, 95)
        
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Clip to [0, 1]
        cam = np.clip(cam, 0, 1)
        
        return cam



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
    return video.unsqueeze(0), frames

# ================== XAI VISUALIZATION UTILITIES ==================

def visualize_audio_gradcam(mfcc, cam_heatmap):
    """
    Create visualization of Grad-CAM on MFCC spectrogram
    
    Returns:
        Base64 encoded image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Enhance CAM contrast
    cam_enhanced = cam_heatmap.copy()
    # Apply power law transformation to enhance highlights
    cam_enhanced = np.power(cam_enhanced, 0.5)  # Makes highlights more visible
    
    # Original MFCC
    axes[0].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original MFCC', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('MFCC Coefficients')
    
    # Grad-CAM heatmap with enhanced contrast
    im1 = axes[1].imshow(cam_enhanced, aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Attention Map', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('MFCC Coefficients')
    plt.colorbar(im1, ax=axes[1], label='Importance')
    
    # Overlay with higher alpha for visibility
    axes[2].imshow(mfcc, aspect='auto', origin='lower', cmap='gray', alpha=0.5)
    im2 = axes[2].imshow(cam_enhanced, aspect='auto', origin='lower', cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title('MFCC + Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('MFCC Coefficients')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def visualize_video_gradcam(frames, cam_3d, num_frames_to_show=8):
    """
    Create visualization of Grad-CAM on video frames
    
    Args:
        frames: Original video frames (T, H, W, C)
        cam_3d: 3D Grad-CAM heatmap (T, H, W) or (H, W)
        num_frames_to_show: Number of frames to visualize
    
    Returns:
        Base64 encoded image
    """
    num_frames = len(frames)
    
    # Adjust num_frames_to_show if we have fewer frames
    num_frames_to_show = min(num_frames_to_show, num_frames)
    
    indices = np.linspace(0, num_frames - 1, num_frames_to_show).astype(int)
    
    fig, axes = plt.subplots(2, num_frames_to_show, figsize=(20, 5))
    
    # Handle case where we only have 1 frame to show
    if num_frames_to_show == 1:
        axes = axes.reshape(2, 1)
    
    for idx, frame_idx in enumerate(indices):
        frame = frames[frame_idx]
        
        # Get the appropriate CAM frame
        if len(cam_3d.shape) == 3 and cam_3d.shape[0] > frame_idx:
            # 3D CAM with temporal dimension
            cam_frame = cam_3d[frame_idx]
        elif len(cam_3d.shape) == 3:
            # 3D CAM but not enough frames, use the last one
            cam_frame = cam_3d[-1]
        elif len(cam_3d.shape) == 2:
            # 2D CAM (spatial only), use for all frames
            cam_frame = cam_3d
        else:
            # Fallback: create empty cam
            cam_frame = np.zeros((frame.shape[0], frame.shape[1]))
        
        # Enhance CAM contrast - apply power law transformation
        cam_enhanced = np.power(cam_frame, 0.6)  # Makes highlights more visible
        
        # Resize cam to match frame size
        cam_resized = cv2.resize(cam_enhanced, (frame.shape[1], frame.shape[0]))
        
        # Apply a threshold to make weak activations transparent
        cam_resized_masked = np.where(cam_resized > 0.3, cam_resized, 0)
        
        # Original frame
        axes[0, idx].imshow(frame)
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_title(f'Frame {frame_idx}\n(Original)', fontsize=10, fontweight='bold')
        else:
            axes[0, idx].set_title(f'Frame {frame_idx}', fontsize=10)
        
        # Overlay with enhanced visibility
        axes[1, idx].imshow(frame)
        # Use masked CAM with higher alpha for better visibility
        im = axes[1, idx].imshow(cam_resized_masked, cmap='jet', alpha=0.6, vmin=0, vmax=1, interpolation='bilinear')
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_title('Grad-CAM\nOverlay', fontsize=10, fontweight='bold')
    
    # Add a colorbar to show the attention scale
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


# ================== API ==================
@app.post("/predict")
async def predict_video(file: UploadFile = File(...), enable_gradcam: bool = True):
    """
    Predict if video is AI-generated or real, with optional Grad-CAM visualization
    
    Args:
        file: Uploaded video file
        enable_gradcam: Whether to generate Grad-CAM visualizations (default: True)
    """
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

        # 5. Audio Grad-CAM
        audio_gradcam_viz = None
        if enable_gradcam:
            try:
                # Target the last conv layer before adaptive pooling
                target_layer = audio_model.features[6]  # Conv2d(64, 128, ...)
                
                audio_gradcam = GradCAMPlusPlus(audio_model, target_layer)
                
                # Create new tensor with gradient enabled for Grad-CAM
                audio_x_grad = audio_x.clone().detach().requires_grad_(True)
                
                # Generate CAM for predicted class
                predicted_class = audio_probs.argmax(dim=1).item()
                cam_audio = audio_gradcam.generate_cam(audio_x_grad, target_class=predicted_class)
                
                # Resize CAM to MFCC size
                cam_resized = cv2.resize(cam_audio, (mfcc.shape[1], mfcc.shape[0]))
                
                # Create visualization
                audio_gradcam_viz = visualize_audio_gradcam(mfcc, cam_resized)
                
                audio_gradcam.remove_hooks()
            except Exception as e:
                print(f"Audio Grad-CAM error: {e}")

        # ================= VIDEO MODEL (SWIN3D) =================
        video_x, original_frames = load_video_for_swin(video_path)
        video_x = video_x.to(DEVICE)

        with torch.no_grad():
            video_probs = torch.softmax(video_model(video_x), dim=1)

        video_real = float(video_probs[0][0])
        video_ai = float(video_probs[0][1])

        # 6. Video Grad-CAM
        video_gradcam_viz = None
        if enable_gradcam:
            try:
                # Target the last layer before the head
                # For Swin3D, we'll target the last block
                target_layer = video_model.features[-1]  # Last feature block
                
                video_gradcam = GradCAMPlusPlus(video_model, target_layer)
                
                # Create new tensor with gradient enabled for Grad-CAM
                video_x_grad = video_x.clone().detach().requires_grad_(True)
                
                # Generate CAM for predicted class
                predicted_class = video_probs.argmax(dim=1).item()
                cam_video = video_gradcam.generate_cam(video_x_grad, target_class=predicted_class)
                
                # The Swin3D CAM is typically 2D spatial (H, W)
                # We replicate it across all frames for visualization
                if len(cam_video.shape) == 2:
                    # (H, W) - this is the spatial attention map
                    cam_video_3d = np.stack([cam_video] * len(original_frames))
                elif len(cam_video.shape) == 3:
                    # If somehow we got 3D, use it directly
                    cam_video_3d = cam_video
                else:
                    # 1D - reshape to spatial
                    side = int(np.sqrt(cam_video.shape[0]))
                    cam_video_2d = cam_video.reshape(side, side)
                    cam_video_3d = np.stack([cam_video_2d] * len(original_frames))
                
                # Create visualization
                video_gradcam_viz = visualize_video_gradcam(original_frames, cam_video_3d)
                
                video_gradcam.remove_hooks()
            except Exception as e:
                print(f"Video Grad-CAM error: {e}")
                import traceback
                traceback.print_exc()

        # ================= FUSION =================
        # Weighted fusion (audio slightly higher)
        final_ai = 0.6 * audio_ai + 0.4 * video_ai
        final_real = 1.0 - final_ai

        prediction = "AI-GENERATED" if final_ai >= 0.5 else "REAL"
        confidence = round(max(final_ai, final_real) * 100, 2)

        response = {
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

        # Add Grad-CAM visualizations if enabled
        if enable_gradcam:
            response["explainability"] = {
                "audio_gradcam": audio_gradcam_viz,
                "video_gradcam": video_gradcam_viz
            }

        return response

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


@app.post("/predict_with_xai")
async def predict_video_with_full_xai(file: UploadFile = File(...)):
    """
    Enhanced prediction endpoint with detailed XAI analysis
    
    Returns predictions with:
    - Grad-CAM visualizations
    - Attention maps
    - Feature importance scores
    """
    return await predict_video(file, enable_gradcam=True)


def predict_video_visual(video_path):
    x, frames = load_video_for_swin(video_path)
    x = x.to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(video_model(x), dim=1)

    return {
        "real_probability": float(probs[0][0]),
        "ai_probability": float(probs[0][1])
    }


# ================== ADDITIONAL XAI ENDPOINTS ==================

@app.post("/explain_audio")
async def explain_audio_prediction(file: UploadFile = File(...)):
    """
    Generate detailed explanation for audio-based prediction
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        audio_path = temp_audio.name
    
    try:
        mfcc = extract_mfcc(audio_path)
        audio_x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            probs = torch.softmax(audio_model(audio_x), dim=1)
        
        # Grad-CAM
        target_layer = audio_model.features[6]
        gradcam = GradCAMPlusPlus(audio_model, target_layer)
        
        # Create new tensor with gradient enabled
        audio_x_grad = audio_x.clone().detach().requires_grad_(True)
        
        cam = gradcam.generate_cam(audio_x_grad, target_class=probs.argmax(dim=1).item())
        cam_resized = cv2.resize(cam, (mfcc.shape[1], mfcc.shape[0]))
        
        viz = visualize_audio_gradcam(mfcc, cam_resized)
        
        gradcam.remove_hooks()
        
        return {
            "prediction": "AI-GENERATED" if probs[0][1] > 0.5 else "REAL",
            "probabilities": {
                "real": float(probs[0][0]),
                "ai": float(probs[0][1])
            },
            "gradcam_visualization": viz
        }
    
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/explain_video")
async def explain_video_prediction(file: UploadFile = File(...)):
    """
    Generate detailed explanation for video-based prediction
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name
    
    try:
        video_x, frames = load_video_for_swin(video_path)
        video_x = video_x.to(DEVICE)
        
        with torch.no_grad():
            probs = torch.softmax(video_model(video_x), dim=1)
        
        # Grad-CAM
        target_layer = video_model.features[-1]
        gradcam = GradCAMPlusPlus(video_model, target_layer)
        
        # Create new tensor with gradient enabled
        video_x_grad = video_x.clone().detach().requires_grad_(True)
        
        cam = gradcam.generate_cam(video_x_grad, target_class=probs.argmax(dim=1).item())
        
        if len(cam.shape) == 2:
            cam_3d = np.stack([cam] * len(frames))
        else:
            cam_3d = cam
        
        viz = visualize_video_gradcam(frames, cam_3d)
        
        gradcam.remove_hooks()
        
        return {
            "prediction": "AI-GENERATED" if probs[0][1] > 0.5 else "REAL",
            "probabilities": {
                "real": float(probs[0][0]),
                "ai": float(probs[0][1])
            },
            "gradcam_visualization": viz
        }
    
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)