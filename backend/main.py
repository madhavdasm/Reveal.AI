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

# Import pytorch-grad-cam library
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
    torch.load("mfcc_audio_detector.pth", map_location=DEVICE, weights_only=False)
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

# ================== VIDEO MODEL ==================
video_model = swin3d_t(weights=None)
video_model.head = torch.nn.Linear(video_model.head.in_features, 2)

video_model.load_state_dict(
    torch.load("best_swin_model.pth", map_location=DEVICE, weights_only=False)
)

video_model.to(DEVICE)
video_model.eval()

# ================== ATTENTION ROLLOUT FOR SWIN3D ==================

ATTN_MAPS = []

def save_attention(module, input, output):
    try:
        if isinstance(output, tuple):
            output = output[0]

        if hasattr(output, "shape") and len(output.shape) == 4:
            # Expecting (B, heads, N, N)
            ATTN_MAPS.append(output.detach().cpu())
    except:
        pass



def register_attention_hooks(model):
    hooks = []

    for name, module in model.named_modules():
        # Swin3D attention layer name pattern
        if "attn" in name.lower() and hasattr(module, "forward"):
            
            def hook_fn(module, input, output):
                try:
                    # Swin attention output is (B, T*H*W, C) or similar
                    if isinstance(output, tuple):
                        output = output[0]

                    if hasattr(output, "shape"):
                        ATTN_MAPS.append(output.detach().cpu())
                except:
                    pass

            hooks.append(module.register_forward_hook(hook_fn))

    return hooks



def compute_attention_rollout(attn_maps):
    if len(attn_maps) == 0:
        return None

    attn = attn_maps[-1]

    # (B, T, H, W, C)
    if len(attn.shape) == 5:
        attn = attn.mean(dim=4)   # (B, T, H, W)
        attn = attn[0]            # (T, H, W)

        # 🔥 NORMALIZE PER FRAME
        attn = attn - attn.min()
        attn = attn / (attn.max() + 1e-8)

        return attn.numpy()

    if len(attn.shape) == 3:
        attn = attn.mean(dim=2)
        attn = attn - attn.min()
        attn = attn / (attn.max() + 1e-8)
        return attn.numpy()

    return None



def tokens_to_spatial_map(rollout, num_frames):
    # If already (T, H, W) → return directly
    if len(rollout.shape) == 3:
        return rollout

    # Old fallback
    rollout = rollout[0]
    N = rollout.shape[0]

    spatial = int(np.sqrt(N / num_frames))
    spatial = max(spatial, 1)

    total_needed = num_frames * spatial * spatial
    rollout = rollout[:total_needed]

    attn_map = rollout.reshape(num_frames, spatial, spatial)
    return attn_map

def find_swin3d_target_layers(model):
    """
    Automatically find the best target layer(s) for Swin3D Grad-CAM
    
    Returns:
        list: Target layers suitable for Grad-CAM
    """
    print("\n=== Analyzing Swin3D Architecture ===")
    
    target_layers = []
    
    # Strategy 1: Look for the last norm layer in features
    if hasattr(model, 'features'):
        features = model.features
        print(f"Features has {len(features)} modules")
        
        # Walk through the features in reverse to find norm layers
        for i in range(len(features) - 1, -1, -1):
            module = features[i]
            module_name = f"features[{i}]"
            print(f"  {module_name}: {type(module).__name__}")
            
            # Check if it's a Sequential containing blocks
            if isinstance(module, nn.Sequential):
                for j in range(len(module) - 1, -1, -1):
                    submodule = module[j]
                    submodule_name = f"features[{i}][{j}]"
                    print(f"    {submodule_name}: {type(submodule).__name__}")
                    
                    # Look for blocks with norm layers
                    if hasattr(submodule, 'blocks'):
                        # Found blocks! Now find norm layers
                        for k in range(len(submodule.blocks) - 1, -1, -1):
                            block = submodule.blocks[k]
                            if hasattr(block, 'norm2'):
                                target_layer = block.norm2
                                target_layers.append(target_layer)
                                print(f"    ✓ Found target: features[{i}][{j}].blocks[{k}].norm2")
                                return target_layers
                            elif hasattr(block, 'norm1'):
                                target_layer = block.norm1
                                target_layers.append(target_layer)
                                print(f"    ✓ Found target: features[{i}][{j}].blocks[{k}].norm1")
                                return target_layers
                    
                    # Check if the submodule itself is a norm layer
                    if isinstance(submodule, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm)):
                        target_layers.append(submodule)
                        print(f"    ✓ Found target: {submodule_name}")
                        return target_layers
            
            # Check if it's directly a norm layer or has norm
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm)):
                target_layers.append(module)
                print(f"  ✓ Found target: {module_name}")
                return target_layers
            
            if hasattr(module, 'norm'):
                target_layers.append(module.norm)
                print(f"  ✓ Found target: {module_name}.norm")
                return target_layers
    
    # Fallback: use the last layer of features
    if not target_layers and hasattr(model, 'features'):
        print("  ⚠ Using fallback: features[-1]")
        target_layers = [model.features[-1]]
    
    print("=== End Architecture Analysis ===\n")
    return target_layers


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


# ================== SWIN3D RESHAPE TRANSFORM ==================
def swin3d_reshape_transform(tensor, height=7, width=7, depth=8):
    """
    Reshape transform for Swin3D Transformer
    Handles various output formats from Swin3D layers
    """
    # Handle different tensor shapes
    if len(tensor.shape) == 3:
        # (B, N, C) format - typical Swin output
        B, N, C = tensor.shape
        
        # Calculate dimensions
        expected_n = depth * height * width
        
        if N == expected_n:
            # Perfect match
            result = tensor.reshape(B, depth, height, width, C)
        elif N == expected_n + 1:
            # Has class token, remove it
            tensor = tensor[:, 1:, :]
            result = tensor.reshape(B, depth, height, width, C)
        else:
            # Try to infer dimensions
            # Common: N = H*W (no temporal) or N = T*H*W
            if N == height * width:
                # 2D spatial only, add temporal dimension
                result = tensor.reshape(B, height, width, C).unsqueeze(1)
                result = result.expand(B, depth, height, width, C)
            else:
                # Best effort: assume square spatial
                spatial_size = int(np.sqrt(N / depth))
                result = tensor.reshape(B, depth, spatial_size, spatial_size, C)
        
        # Permute to (B, C, T, H, W)
        result = result.permute(0, 4, 1, 2, 3)
        
    elif len(tensor.shape) == 5:
        # Already in (B, C, T, H, W) or (B, T, H, W, C) format
        if tensor.shape[1] < tensor.shape[-1]:
            # Likely (B, T, H, W, C) - need to permute
            result = tensor.permute(0, 4, 1, 2, 3)
        else:
            # Already (B, C, T, H, W)
            result = tensor
    else:
        # Fallback - return as is
        print(f"Warning: Unexpected tensor shape {tensor.shape}")
        result = tensor
    
    return result


# ================== XAI VISUALIZATION UTILITIES ==================

def visualize_audio_gradcam(mfcc, cam_heatmap):
    """
    Create visualization of Grad-CAM on MFCC spectrogram
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    cam_enhanced = cam_heatmap.copy()
    cam_enhanced = np.power(cam_enhanced, 0.5)
    
    # Original MFCC
    axes[0].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original MFCC', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('MFCC Coefficients')
    
    # Grad-CAM heatmap
    im1 = axes[1].imshow(cam_enhanced, aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Attention Map', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('MFCC Coefficients')
    plt.colorbar(im1, ax=axes[1], label='Importance')
    
    # Overlay
    axes[2].imshow(mfcc, aspect='auto', origin='lower', cmap='gray', alpha=0.5)
    axes[2].imshow(cam_enhanced, aspect='auto', origin='lower', cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title('MFCC + Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('MFCC Coefficients')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def visualize_video_gradcam_3d(frames, cam_3d, num_frames_to_show=8):
    """
    Enhanced visualization for 3D video Grad-CAM
    """
    num_frames = len(frames)
    num_frames_to_show = min(num_frames_to_show, num_frames)
    
    indices = np.linspace(0, num_frames - 1, num_frames_to_show).astype(int)
    
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(3, num_frames_to_show, hspace=0.35, wspace=0.1)
    
    for idx, frame_idx in enumerate(indices):
        frame = frames[frame_idx]
        
        # Get corresponding CAM
        if len(cam_3d.shape) == 3 and frame_idx < cam_3d.shape[0]:
            cam_frame = cam_3d[frame_idx]
        elif len(cam_3d.shape) == 3:
            cam_frame = cam_3d[-1]
        elif len(cam_3d.shape) == 2:
            cam_frame = cam_3d
        else:
            cam_frame = np.zeros((frame.shape[0], frame.shape[1]))
        
        # Resize CAM to match frame size
        cam_resized = cv2.resize(cam_frame, (frame.shape[1], frame.shape[0]), 
                                interpolation=cv2.INTER_CUBIC)
        
        # Apply smoothing
        cam_smooth = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=2.5, sigmaY=2.5)
        
        # Threshold
        threshold = 0.35
        cam_thresholded = np.where(cam_smooth > threshold, cam_smooth, 0)
        
        if cam_thresholded.max() > 0:
            cam_thresholded = cam_thresholded / cam_thresholded.max()
        
        # Row 1: Original frames
        ax1 = fig.add_subplot(gs[0, idx])
        ax1.imshow(frame)
        ax1.axis('off')
        if idx == 0:
            ax1.set_title('Original Frames', fontsize=11, fontweight='bold', pad=10)
        
        # Row 2: Pure heatmap
        ax2 = fig.add_subplot(gs[1, idx])
        im = ax2.imshow(cam_smooth, cmap='jet', vmin=0, vmax=1)
        ax2.axis('off')
        if idx == 0:
            ax2.set_title('Attention Heatmap\n(Spatial Focus)', fontsize=11, fontweight='bold', pad=10)
        
        # Row 3: Overlay
        ax3 = fig.add_subplot(gs[2, idx])
        ax3.imshow(frame)
        ax3.imshow(cam_thresholded, cmap='jet', alpha=0.55, vmin=0, vmax=1, 
                  interpolation='bilinear')
        ax3.axis('off')
        if idx == 0:
            ax3.set_title('AI Region Detection\n(Mouth, Fingers, etc.)', fontsize=11, fontweight='bold', pad=10)
        
        ax3.text(0.5, -0.15, f'Frame {frame_idx}', transform=ax3.transAxes,
                ha='center', va='top', fontsize=9, color='black')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=20, fontsize=11)
    
    # Title
    fig.suptitle('Video Grad-CAM: Spatial-Temporal Analysis\n(Red areas = Model\'s focus for AI detection)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


# ================== API ==================
@app.post("/predict")
async def predict_video(file: UploadFile = File(...), enable_gradcam: bool = True):
    """
    Predict if video is AI-generated or real, with Grad-CAM visualization
    """
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    try:
        # ================= AUDIO MODEL =================
        extract_audio_from_video(video_path, temp_audio)
        mfcc = extract_mfcc(temp_audio)
        
        audio_x = (
            torch.tensor(mfcc)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        with torch.no_grad():
            audio_probs = torch.softmax(audio_model(audio_x), dim=1)

        audio_real = float(audio_probs[0][0])
        audio_ai = float(audio_probs[0][1])

        # Audio Grad-CAM
        audio_gradcam_viz = None
        if enable_gradcam:
            try:
                target_layer = [audio_model.features[6]]
                predicted_class = audio_probs.argmax(dim=1).item()
                targets = [ClassifierOutputTarget(predicted_class)]
                
                with GradCAMPlusPlus(model=audio_model, target_layers=target_layer) as cam:
                    grayscale_cam = cam(input_tensor=audio_x, targets=targets, aug_smooth=True)
                    grayscale_cam = grayscale_cam[0, :]
                
                cam_resized = cv2.resize(grayscale_cam, (mfcc.shape[1], mfcc.shape[0]))
                audio_gradcam_viz = visualize_audio_gradcam(mfcc, cam_resized)
                
            except Exception as e:
                print(f"Audio Grad-CAM error: {e}")
                import traceback
                traceback.print_exc()

        # ================= VIDEO MODEL (SWIN3D) =================
        video_x, original_frames = load_video_for_swin(video_path)
        video_x = video_x.to(DEVICE)

        # Register hooks BEFORE forward pass
        ATTN_MAPS.clear()
        hooks = register_attention_hooks(video_model)

        with torch.no_grad():
            video_logits = video_model(video_x)
            video_probs = torch.softmax(video_logits, dim=1)

        # Remove hooks
        for h in hooks:
            h.remove()
        
        print("Captured attention maps:", len(ATTN_MAPS))
        if len(ATTN_MAPS) > 0:
            print("Sample attention map shape:", ATTN_MAPS[0].shape)    


        video_real = float(video_probs[0][0])
        video_ai = float(video_probs[0][1])

        # ================= VIDEO XAI USING ATTENTION ROLLOUT =================
        video_gradcam_viz = None
        if enable_gradcam:
            try:
                rollout = compute_attention_rollout(ATTN_MAPS)
                print("Rollout shape:", None if rollout is None else rollout.shape)


                if rollout is not None:
                    cam_3d = tokens_to_spatial_map(rollout, len(original_frames))
                    video_gradcam_viz = visualize_video_gradcam_3d(original_frames, cam_3d)

            except Exception as e:
                print(f"Attention Rollout error: {e}")


        # ================= FUSION =================
        final_ai = 0.6 * audio_ai + 0.4 * video_ai
        final_real = 1.0 - final_ai

        prediction = "AI-GENERATED" if final_ai >= 0.5 else "REAL"
        confidence = round(max(final_ai, final_real) * 100, 2)

        response = {
            "prediction": prediction,
            "confidence": confidence,
            "audio_model": {
                "real_probability": round(audio_real, 4),
                "ai_probability": round(audio_ai, 4)
            },
            "video_model": {
                "real_probability": round(video_real, 4),
                "ai_probability": round(video_ai, 4)
            },
            "final_ai_probability": round(final_ai, 4),
            "final_real_probability": round(final_real, 4)
        }

        if enable_gradcam:
            response["explainability"] = {
                "audio_gradcam": audio_gradcam_viz,
                "video_gradcam": video_gradcam_viz
            }
        print("video_gradcam_viz is None?", video_gradcam_viz is None)  # <-- ADD HERE
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


@app.post("/predict_with_xai")
async def predict_video_with_full_xai(file: UploadFile = File(...)):
    """Enhanced prediction with XAI"""
    return await predict_video(file, enable_gradcam=True)


@app.post("/explain_audio")
async def explain_audio_prediction(file: UploadFile = File(...)):
    """Generate audio explanation"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        audio_path = temp_audio.name
    
    try:
        mfcc = extract_mfcc(audio_path)
        audio_x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            probs = torch.softmax(audio_model(audio_x), dim=1)
        
        target_layer = [audio_model.features[6]]
        targets = [ClassifierOutputTarget(probs.argmax(dim=1).item())]
        
        with GradCAMPlusPlus(model=audio_model, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=audio_x, targets=targets, aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
        
        cam_resized = cv2.resize(grayscale_cam, (mfcc.shape[1], mfcc.shape[0]))
        viz = visualize_audio_gradcam(mfcc, cam_resized)
        
        return {
            "prediction": "AI-GENERATED" if probs[0][1] > 0.5 else "REAL",
            "probabilities": {"real": float(probs[0][0]), "ai": float(probs[0][1])},
            "gradcam_visualization": viz
        }
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/explain_video")
async def explain_video_prediction(file: UploadFile = File(...)):
    """Generate video explanation"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        video_path = temp_video.name
    
    try:
        video_x, frames = load_video_for_swin(video_path)
        video_x = video_x.to(DEVICE)

        ATTN_MAPS.clear()
        hooks = register_attention_hooks(video_model)

        with torch.no_grad():
            logits = video_model(video_x)
            probs = torch.softmax(logits, dim=1)

        for h in hooks:
            h.remove()

        rollout = compute_attention_rollout(ATTN_MAPS)

        cam_3d = tokens_to_spatial_map(rollout, len(frames))
        viz = visualize_video_gradcam_3d(frames, cam_3d)

        return {
            "prediction": "AI-GENERATED" if probs[0][1] > 0.5 else "REAL",
            "probabilities": {"real": float(probs[0][0]), "ai": float(probs[0][1])},
            "gradcam_visualization": viz
        }

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)