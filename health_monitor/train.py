"""
Stereo Health Monitor — Model + Training

Architecture:
  Per frame:
    left_img, right_img → ResNet18 → left_emb, right_emb (512 each)
    diff_emb = left_emb - right_emb
    frame_emb = concat(left_emb, right_emb, diff_emb) → Linear(1536→128)

  Sequence:
    20 frame embeddings (20, 128) → GRU(128, 64) → last hidden (64,)

  Classifier:
    Linear(64→32) + ReLU + Dropout → Linear(32→1) + Sigmoid → severity
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # non-GUI backend — saves to file instead of displaying
import matplotlib.pyplot as plt

from stereo_health_dataset import StereoHealthDataset
from config import DATASET_ROOT, CHECKPOINT_DIR, SEED

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE    = 16
EPOCHS        = 30
LR_BACKBONE   = 1e-5
LR_REST       = 1e-3
VAL_SPLIT     = 0.2
CROSSOVER_THR = 0.5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUICK_SAMPLES = 100

# ── Model ─────────────────────────────────────────────────────────────────────
class FrameEncoder(nn.Module):
    """
    Encodes a (left, right) image pair into a single frame embedding.
    Shared ResNet18 backbone for both cameras.
    Difference embedding explicitly captures LR asymmetry.
    """
    def __init__(self, embed_dim=128, freeze_backbone=True):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # unfreeze the last ResNet layer (layer4) for fine-tuning
            for param in list(self.backbone.children())[-1].parameters():
                param.requires_grad = True

        self.project = nn.Sequential(
            nn.Linear(1536, embed_dim),
            nn.ReLU(),
        )

    def encode_image(self, x):
        """x: (B, 1, H, W) → (B, 512)"""
        x    = x.repeat(1, 3, 1, 1)       # grayscale → 3-channel
        feat = self.backbone(x)            # (B, 512, 1, 1)
        return feat.flatten(1)            # (B, 512)

    def forward(self, cam0, cam1):
        """cam0, cam1: (B, 1, H, W) → (B, embed_dim)"""
        left_emb  = self.encode_image(cam0)
        right_emb = self.encode_image(cam1)
        diff_emb  = left_emb - right_emb
        combined  = torch.cat([left_emb, right_emb, diff_emb], dim=1)
        return self.project(combined)


class StereoHealthMonitor(nn.Module):
    """
    Input:
      cam0: (B, T, 1, H, W)  — T clean left frames
      cam1: (B, T, 1, H, W)  — T degraded right frames
    Output:
      severity: (B,) in [0, 1]
    """
    def __init__(self, embed_dim=128, gru_hidden=64,
                 dropout=0.3, freeze_backbone=True):
        super().__init__()

        self.encoder = FrameEncoder(embed_dim, freeze_backbone)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, cam0, cam1):
        B, T, C, H, W = cam0.shape

        cam0_flat  = cam0.view(B * T, C, H, W)
        cam1_flat  = cam1.view(B * T, C, H, W)
        frame_embs = self.encoder(cam0_flat, cam1_flat)  # (B*T, embed_dim)
        frame_embs = frame_embs.view(B, T, -1)           # (B, T, embed_dim)

        _, hidden  = self.gru(frame_embs)                # (1, B, gru_hidden)
        hidden     = hidden.squeeze(0)                   # (B, gru_hidden)

        return self.classifier(hidden).squeeze(1)        # (B,)

# ── Metrics ───────────────────────────────────────────────────────────────────
def crossover_accuracy(preds, targets, thr=CROSSOVER_THR):
    pred_flag   = (preds   >= thr).float()
    target_flag = (targets >= thr).float()
    return (pred_flag == target_flag).float().mean().item()

# ── Train / Eval loops ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = mae_sum = acc_sum = 0

    for cam0, cam1, severity in loader:
        cam0, cam1, severity = cam0.to(device), cam1.to(device), severity.to(device)

        optimizer.zero_grad()
        preds = model(cam0, cam1)
        loss  = criterion(preds, severity)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        mae_sum  += torch.abs(preds - severity).mean().item()
        acc_sum  += crossover_accuracy(preds, severity)

    n = len(loader)
    return loss_sum / n, mae_sum / n, acc_sum / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = mae_sum = acc_sum = 0
    all_preds, all_targets = [], []

    for cam0, cam1, severity in loader:
        cam0, cam1, severity = cam0.to(device), cam1.to(device), severity.to(device)

        preds = model(cam0, cam1)
        loss  = criterion(preds, severity)

        loss_sum += loss.item()
        mae_sum  += torch.abs(preds - severity).mean().item()
        acc_sum  += crossover_accuracy(preds, severity)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(severity.cpu().numpy())

    n = len(loader)
    return (loss_sum / n, mae_sum / n, acc_sum / n,
            np.array(all_preds), np.array(all_targets))

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Stereo Health Monitor — Training", fontsize=12)

    for ax, key, title in zip(
        axes,
        ["loss", "mae", "acc"],
        ["HuberLoss", "MAE", "Crossover Accuracy"]
    ):
        ax.plot(history[f"train_{key}"], label="train")
        ax.plot(history[f"val_{key}"],   label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Saved training_curves.png")


def plot_predictions(preds, targets):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Validation — Predictions vs Ground Truth")

    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], "r--", label="perfect")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("True severity")
    ax.set_ylabel("Predicted severity")
    ax.set_title("Predicted vs True")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    healthy_preds  = preds[targets <  0.5]
    degraded_preds = preds[targets >= 0.5]
    ax.hist(healthy_preds,  bins=30, alpha=0.6, label="healthy",  color="steelblue")
    ax.hist(degraded_preds, bins=30, alpha=0.6, label="degraded", color="crimson")
    ax.axvline(0.5, color="black", linestyle="--", label="threshold=0.5")
    ax.set_xlabel("Predicted severity")
    ax.set_title("Prediction distribution by class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("val_predictions.png", dpi=150)
    plt.close()
    print("Saved val_predictions.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Subsample to 100 windows for a quick pipeline test")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # data
    dataset = StereoHealthDataset(dataset_root=DATASET_ROOT)
    print(f"Dataset: {len(dataset)} windows")

    # subsample for quick test
    if args.quick:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(dataset), size=min(QUICK_SAMPLES, len(dataset)), replace=False)
        dataset = torch.utils.data.Subset(dataset, idx.tolist())
        print(f"QUICK TEST MODE — using {len(dataset)} windows")

    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    n_workers = 0 if args.quick else 4

    # weighted sampler — balance healthy vs degraded to fix mid-range bias
    train_severities = np.array([dataset.windows[i][1]
                                  for i in train_set.indices])
    bins    = [0.0, 0.25, 0.5, 0.75, 1.01]
    bin_idx = np.digitize(train_severities, bins) - 1
    bin_counts = np.bincount(bin_idx, minlength=len(bins) - 1).astype(float)
    bin_weights = 1.0 / np.maximum(bin_counts, 1)
    sample_weights = torch.tensor(bin_weights[bin_idx], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=n_workers, pin_memory=True)
    print(f"Train: {len(train_set)}  |  Val: {len(val_set)}")
    print(f"Bin counts (healthy/mild/degraded/severe): {bin_counts.astype(int)}")

    # model
    model = StereoHealthMonitor(
        embed_dim=128, gru_hidden=64,
        dropout=0.3, freeze_backbone=True,
    ).to(DEVICE)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total  |  {trainable:,} trainable")

    # two param groups — lower LR for backbone
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    other_params    = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": other_params,    "lr": LR_REST},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.HuberLoss(delta=0.1)

    # training
    history      = {k: [] for k in ["train_loss", "train_mae", "train_acc",
                                     "val_loss",   "val_mae",   "val_acc"]}
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_mae, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_mae, val_acc, val_preds, val_targets = evaluate(
            model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        for k, v in zip(
            ["train_loss", "train_mae", "train_acc",
             "val_loss",   "val_mae",   "val_acc"],
            [tr_loss, tr_mae, tr_acc, val_loss, val_mae, val_acc]
        ):
            history[k].append(v)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss {tr_loss:.4f}/{val_loss:.4f} | "
              f"MAE {tr_mae:.4f}/{val_mae:.4f} | "
              f"Acc {tr_acc:.3f}/{val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            print(f"  ✓ best model saved (val_loss={val_loss:.4f})")

    # plots
    plot_training(history)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt"))
    _, _, _, val_preds, val_targets = evaluate(model, val_loader, criterion, DEVICE)
    plot_predictions(val_preds, val_targets)

    print(f"\nBest val loss      : {best_val_loss:.4f}")
    print(f"Final val accuracy : {val_acc:.3f}")


if __name__ == "__main__":
    main()