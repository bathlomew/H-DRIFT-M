import os
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import convolve  # Added for flexible kernel sizes

import matplotlib.pyplot as plt

# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    # Ensure this directory exists and contains your "gold" images
    data_dir: str = "data/train/good" 
    image_size: int = 224
    batch_size: int = 8
    epochs: int = 15
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    save_dir: str = "outputs_h_drift_m"

    # Degradation ranges for the Virtual Sensor
    defocus_sigma: Tuple[float, float] = (0.0, 3.0) # For dz
    illum_scale: Tuple[float, float] = (0.6, 1.1)   # For dI
    noise_std: Tuple[float, float] = (0.0, 0.05)

    # Structural/Knife Smear ranges
    motion_len_min: int = 3
    motion_len_max: int = 15

# -------------------------
# Corrected Motion Blur (Fixed ValueError)
# -------------------------
def motion_blur_np(img: Image.Image, length: int, angle_deg: float) -> Image.Image:
    """
    Uses scipy.ndimage.convolve to handle arbitrary kernel sizes, 
    solving the PIL 'bad kernel size' error.
    """
    if length < 3:
        return img

    # Create a horizontal line kernel
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    
    # Rotate kernel using PIL to the specific drift angle
    ker_img = Image.fromarray((kernel * 255).astype(np.uint8), mode="L")
    ker_img = ker_img.rotate(angle_deg, resample=Image.BILINEAR)
    ker = np.array(ker_img).astype(np.float32)
    
    # Normalize kernel
    if ker.sum() <= 0:
        return img
    ker /= ker.sum()

    # Apply convolution to RGB channels
    img_array = np.array(img).astype(np.float32)
    channels = []
    for i in range(3): # Apply to R, G, and B
        channels.append(convolve(img_array[:, :, i], ker))
    
    out_array = np.stack(channels, axis=2)
    out_array = np.clip(out_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(out_array)

# -------------------------
# Degradation + Label Generation
# -------------------------
def degrade_and_label(img: Image.Image, cfg: Config):
    # Randomly sample drift magnitudes
    dz = random.uniform(*cfg.defocus_sigma)
    dI = random.uniform(*cfg.illum_scale)
    dK = random.uniform(0.0, 1.0) # Probability/Intensity of structural drift

    out = img

    # 1. Axial Defocus (dz)
    if dz > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=dz))

    # 2. Illumination Decay (dI)
    out = ImageEnhance.Brightness(out).enhance(dI)

    # 3. Structural / Knife Smear (dK)
    if random.random() < dK:
        length = random.randint(cfg.motion_len_min, cfg.motion_len_max)
        angle = random.uniform(0, 180)
        out = motion_blur_np(out, length, angle)

    # 4. Sensor Noise
    n_std = random.uniform(*cfg.noise_std)
    if n_std > 0:
        arr = np.array(out).astype(np.float32) / 255.0
        arr += np.random.normal(0, n_std, arr.shape)
        arr = np.clip(arr, 0, 1)
        out = Image.fromarray((arr * 255).astype(np.uint8))

    # Label y = [defocus, illumination_loss, structural_smear]
    y = np.array([dz, 1.0 - dI, dK], dtype=np.float32)
    return out, y

# -------------------------
# Dataset
# -------------------------
class DriftDataset(Dataset):
    def __init__(self, root: str, cfg: Config, tfm):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory {root} not found. Please add 'good' images.")
        self.paths = [os.path.join(root, f) for f in os.listdir(root)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.cfg = cfg
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_deg, y = degrade_and_label(img, self.cfg)
        return self.tfm(img_deg), torch.tensor(y)

# -------------------------
# H-DRIFT-M Model Architecture
# -------------------------
class HDRIFTM(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18 serves as the CNN feature extractor
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        # Regression head outputting 3 drift values
        self.head = nn.Linear(512, 3) 

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.head(f)

# -------------------------
# Execution
# -------------------------
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    seed_everything()

    # Standard ImageNet normalization
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    ds = DriftDataset(cfg.data_dir, cfg, tfm)
    loader = DataLoader(ds, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=cfg.num_workers)

    model = HDRIFTM().to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss() # Minimizing the error between predicted and true drift

    print(f"Starting H-DRIFT-M training on {cfg.device}...")

    for ep in range(cfg.epochs):
        model.train()
        loss_sum = 0

        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        print(f"Epoch {ep+1:02d}/{cfg.epochs} | Avg MSE Loss: {loss_sum/len(loader):.6f}")

    # Save for integration into main(1).py
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "h_drift_m_v_sensor.pt"))
    print(f"Model saved to {cfg.save_dir}/h_drift_m_v_sensor.pt")

    # -------------------------
    # Regression Performance Visualization
    # -------------------------
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.device)
            preds.append(model(x).cpu().numpy())
            gts.append(y.numpy())

    P = np.concatenate(preds)
    G = np.concatenate(gts)
    names = ["Axial Drift (dz)", "Illumination Drift (dI)", "Structural Drift (dK)"]

    for i in range(3):
        plt.figure(figsize=(6, 5))
        plt.scatter(G[:, i], P[:, i], alpha=0.4, color='blue')
        plt.plot([G[:, i].min(), G[:, i].max()], [G[:, i].min(), G[:, i].max()], 'r--')
        plt.xlabel("True Drift Magnitude")
        plt.ylabel("Predicted Drift (H-DRIFT-M)")
        plt.title(f"Metrology Validation: {names[i]}")
        plt.grid(True)
        plt.savefig(os.path.join(cfg.save_dir, f"drift_scatter_{i}.png"))
        plt.close()

if __name__ == "__main__":
    main()