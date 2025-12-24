import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import convolve

import matplotlib.pyplot as plt

# -------------------------
# Config: Tier 1 & 2 Master Upgrades
# -------------------------
@dataclass
class Config:
    # 1. Update this to your high-quality nerve image folder
    data_dir: str = "data/train/good" 
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 50  # Increased for deeper head convergence
    lr: float = 1e-4  # Slower for regression precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "outputs_h_drift_m_v2"

    # 2. Expanded Ranges (Metrology Limits)
    defocus_max: float = 6.0   # dz: Up to 6.0 pixels (Extreme blur)
    illum_min: float = 0.5     # dI: Up to 50% light loss
    motion_max: int = 25       # dK: Up to 25 pixel knife smear

# -------------------------
# Physics-Based Degradation
# -------------------------
def apply_motion_blur(img: Image.Image, length: int, angle_deg: float) -> Image.Image:
    """Simulates knife/microtome smear using custom convolution."""
    if length < 3: return img
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    ker_img = Image.fromarray((kernel * 255).astype(np.uint8), mode="L")
    ker_img = ker_img.rotate(angle_deg, resample=Image.BILINEAR)
    ker = np.array(ker_img).astype(np.float32)
    if ker.sum() <= 0: return img
    ker /= ker.sum()
    img_array = np.array(img).astype(np.float32)
    out_channels = [convolve(img_array[:, :, i], ker) for i in range(3)]
    return Image.fromarray(np.stack(out_channels, axis=2).clip(0, 255).astype(np.uint8))

def degrade_and_normalize(img: Image.Image, cfg: Config):
    """
    Creates the 'Label Normalized' training pair.
    dz_raw (0-6) -> dz_norm (0-1)
    dI_raw (1.0-0.5) -> dI_norm (0-1)
    dK_prob (0-1) -> dK_norm (0-1)
    """
    # Sampling raw drift values
    dz_raw = random.uniform(0.0, cfg.defocus_max)
    dI_raw = random.uniform(cfg.illum_min, 1.0)
    dK_prob = random.uniform(0.0, 1.0) 

    out = img
    # Apply Blur
    if dz_raw > 0.1:
        out = out.filter(ImageFilter.GaussianBlur(radius=dz_raw))
    # Apply Lighting Loss
    out = ImageEnhance.Brightness(out).enhance(dI_raw)
    # Apply Structural Smear
    if dK_prob > 0.2: 
        length = int(dK_prob * cfg.motion_max) 
        out = apply_motion_blur(out, length, random.uniform(0, 180))

    # --- NORMALIZE TARGETS TO 0-1 RANGE ---
    dz_norm = dz_raw / cfg.defocus_max
    dI_norm = (1.0 - dI_raw) / (1.0 - cfg.illum_min)
    dK_norm = dK_prob 

    y = np.array([dz_norm, dI_norm, dK_norm], dtype=np.float32)
    return out, y

# -------------------------
# Dataset & Model
# -------------------------
class DriftDataset(Dataset):
    def __init__(self, root, cfg, tfm):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.cfg, self.tfm = cfg, tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_deg, y = degrade_and_normalize(img, self.cfg)
        return self.tfm(img_deg), torch.tensor(y)

class HDRIFTM(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN Feature Extractor
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        
        # Deeper Regression Head (PhD Upgrade)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting to specific nerve samples
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3) 
        )

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.head(f)

# -------------------------
# Main Execution
# -------------------------
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    loader = DataLoader(DriftDataset(cfg.data_dir, cfg, tfm), batch_size=cfg.batch_size, shuffle=True)
    model = HDRIFTM().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.SmoothL1Loss() # Robust Huber Loss

    print(f"Starting H-DRIFT-M Optimization on {cfg.device}...")

    # --- Training Loop ---
    for ep in range(cfg.epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            pred = model(x)
            
            # Weighted Loss: Prioritize Focus(dz) and Smear(dK)
            loss = (2.0 * loss_fn(pred[:, 0], y[:, 0])) + \
                   (1.0 * loss_fn(pred[:, 1], y[:, 1])) + \
                   (2.0 * loss_fn(pred[:, 2], y[:, 2]))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1:02d}/{cfg.epochs} | Loss: {total_loss/len(loader):.6f}")

    # Save finalized model
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "h_drift_m_v2.pt"))

    # --- AUTO-GENERATION OF SCATTER PLOTS ---
    print("Generating Calibration Curves...")
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.device)
            all_preds.append(model(x).cpu().numpy())
            all_gts.append(y.numpy())

    P, G = np.concatenate(all_preds), np.concatenate(all_gts)
    names = ["Axial Drift (dz)", "Illumination (dI)", "Structural Smear (dK)"]
    
    for i in range(3):
        plt.figure(figsize=(6, 6))
        plt.scatter(G[:, i], P[:, i], alpha=0.3, s=10, label='Predicted Points')
        plt.plot([0, 1], [0, 1], 'r--', label='Ideal Sensor Line') # Perfect Metrology
        plt.xlabel("Ground Truth (Normalized)")
        plt.ylabel("H-DRIFT-M Estimate")
        plt.title(f"Metrology Validation: {names[i]}")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join(cfg.save_dir, f"metrology_scatter_{i}.png"), dpi=300)
        plt.close()
    
    print(f"Validation complete. Final graphs saved in {cfg.save_dir}")

if __name__ == "__main__":
    main()