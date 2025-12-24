import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.transforms.functional as TF


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    data_dir: str = "data"
    model_name: str = "resnet18"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_model: str = "quality_gate_best.pt"
    out_dir: str = "outputs"


# -------------------------
# Model builder
# -------------------------
def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError("Unsupported model")


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    tp = fp = fn = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {"acc": acc, "f1_good": f1}


# -------------------------
# Visualization (SAVE)
# -------------------------
def save_confusion_matrix(model, loader, device, class_names, save_path):
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("H-DRIFT-M Quality Gate – Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_prediction_examples(model, loader, device, save_path, n=8):
    model.eval()
    images_shown = 0

    fig = plt.figure(figsize=(12, 6))

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1)

            for i in range(x.size(0)):
                if images_shown >= n:
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=300)
                    plt.close()
                    return

                img = TF.to_pil_image(x[i].cpu())
                true = "good" if y[i] == 1 else "bad"
                pred = "good" if preds[i] == 1 else "bad"

                plt.subplot(2, n // 2, images_shown + 1)
                plt.imshow(img)
                plt.title(f"T:{true} / P:{pred}")
                plt.axis("off")

                images_shown += 1


def save_confidence_histogram(model, loader, device, save_path):
    probs_good = []
    labels = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)[:, 1]
            probs_good.extend(probs.cpu().numpy())
            labels.extend(y.numpy())

    plt.figure(figsize=(7, 5))
    plt.hist([p for p, l in zip(probs_good, labels) if l == 1],
             bins=20, alpha=0.7, label="Good images")
    plt.hist([p for p, l in zip(probs_good, labels) if l == 0],
             bins=20, alpha=0.7, label="Bad images")

    plt.xlabel("Predicted Probability of 'Good'")
    plt.ylabel("Count")
    plt.title("H-DRIFT-M Quality Confidence Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    cfg = TrainConfig()
    seed_everything()

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(9, sigma=(0.5, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # Datasets
    train_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "val"), transform=val_tf)
    test_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "test"), transform=val_tf)

    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = build_model(cfg.model_name, 2).to(cfg.device)

    labels = [y for _, y in train_ds.samples]
    counts = np.bincount(labels)
    weights = counts.sum() / np.maximum(counts, 1)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=cfg.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    best_f1 = -1.0

    # Training
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)

        scheduler.step()
        train_loss = loss_sum / len(train_ds)
        val_metrics = evaluate(model, val_loader, cfg.device)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_f1_good={val_metrics['f1_good']:.4f}"
        )

        if val_metrics["f1_good"] > best_f1:
            best_f1 = val_metrics["f1_good"]
            torch.save(model.state_dict(), cfg.save_model)

    print("Training complete. Best model saved.")

    # -------------------------
    # Visualization (SAVED)
    # -------------------------
    model.load_state_dict(torch.load(cfg.save_model, map_location=cfg.device))
    model.eval()

    class_names = list(train_ds.class_to_idx.keys())

    save_confusion_matrix(
        model, test_loader, cfg.device, class_names,
        os.path.join(cfg.out_dir, "confusion_matrix.png")
    )

    save_prediction_examples(
        model, test_loader, cfg.device,
        os.path.join(cfg.out_dir, "prediction_examples.png"), n=8
    )

    save_confidence_histogram(
        model, test_loader, cfg.device,
        os.path.join(cfg.out_dir, "confidence_distribution.png")
    )


if __name__ == "__main__":
    main()
