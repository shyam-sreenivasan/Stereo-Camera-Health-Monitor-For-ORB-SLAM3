"""
Model sanity check — run trained StereoHealthMonitor on known-severity windows
from the dataset index and compare predicted vs ground-truth severity.

Displays the first frame of each window (left clean, right degraded) alongside
the predicted and ground-truth severity scores.

Usage:
    python analysis/model_sanity_check.py
    python analysis/model_sanity_check.py --traj traj1 --n-windows 5
"""

import argparse
import sys
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "health_monitor"))

CHECKPOINT  = ROOT / "checkpoints" / "best_model.pt"
DATASET_ROOT = ROOT / "data" / "health_monitor_dataset"
WINDOW_SIZE  = 20
IMG_SIZE     = (224, 224)

# Conditions to probe — spread of severities across degradation types
PROBE_CONDITIONS = [
    "blur_lv1",    # mild Gaussian blur   — low severity
    "blur_lv3",    # moderate Gaussian blur
    "blur_lv5b",   # heavy blur  — approaching cliff
    "blur_lv5c",   # past cliff  — high severity
    "mblur_lv1",   # mild motion blur
    "mblur_lv2",   # moderate motion blur
    "mblur_lv3b",  # heavy motion blur — approaching cliff
    "bright_lv1",  # mild brightness change
    "bright_lv6",  # extreme underexposure
    "snp_lv2",     # light salt & pepper
    "snp_lv5",     # heavy salt & pepper
    "occ_lv2",     # mild occlusion
    "occ_lv5",     # heavy occlusion
]


# ── Model (must match train.py) ────────────────────────────────────────────────

class FrameEncoder(nn.Module):
    def __init__(self, embed_dim=128, freeze_backbone=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.project = nn.Sequential(nn.Linear(1536, embed_dim), nn.ReLU())

    def encode_image(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x).flatten(1)

    def forward(self, cam0, cam1):
        l = self.encode_image(cam0)
        r = self.encode_image(cam1)
        return self.project(torch.cat([l, r, l - r], dim=1))


class StereoHealthMonitor(nn.Module):
    def __init__(self, embed_dim=128, gru_hidden=64, dropout=0.3, freeze_backbone=True):
        super().__init__()
        self.encoder    = FrameEncoder(embed_dim, freeze_backbone)
        self.gru        = nn.GRU(embed_dim, gru_hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, cam0, cam1):
        B, T, C, H, W = cam0.shape
        fe = self.encoder(cam0.view(B*T, C, H, W),
                          cam1.view(B*T, C, H, W)).view(B, T, -1)
        _, h = self.gru(fe)
        return self.classifier(h.squeeze(0)).squeeze(1)


# ── Degradation ────────────────────────────────────────────────────────────────

def apply_config(img, cfg):
    t = cfg.get("type", "none")
    if t in ("none", "clean"):
        return img
    if t == "gaussian_blur":
        ks = int(cfg["kernel_size"])
        if ks % 2 == 0: ks += 1
        return cv2.GaussianBlur(img, (ks, ks), float(cfg.get("sigma", 0)))
    if t == "motion_blur":
        length = int(cfg["length"]); angle = float(cfg.get("angle", 0))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        M = cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1.0)
        return cv2.filter2D(img, -1, cv2.warpAffine(kernel, M, (length, length)))
    if t == "salt_and_pepper":
        density = float(cfg["density"]); out = img.copy()
        n = int(img.size * density)
        coords = np.random.choice(img.size, size=n, replace=False)
        out.flat[coords[:n//2]] = 255; out.flat[coords[n//2:]] = 0
        return out
    if t == "brightness":
        return np.clip(img.astype(np.float32) * float(cfg["alpha"]), 0, 255).astype(np.uint8)
    if t == "occlusion":
        frac = float(cfg["frac"]); h, w = img.shape[:2]
        area = int(h * w * frac); ph = int(np.sqrt(area * h / w))
        pw = int(area / max(ph, 1)); ph, pw = min(ph, h), min(pw, w)
        y = np.random.randint(0, h - ph + 1); x = np.random.randint(0, w - pw + 1)
        out = img.copy(); out[y:y+ph, x:x+pw] = 0
        return out
    raise ValueError(f"Unknown type: {t!r}")


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return cv2.resize(img, IMG_SIZE)


# ── Window loading ─────────────────────────────────────────────────────────────

def load_window(cam0_map, cam1_map, sorted_ts, start_ts, cfg):
    arr       = np.array(sorted_ts)
    start_idx = int(np.argmin(np.abs(arr - start_ts)))
    frame_ts  = sorted_ts[start_idx: start_idx + WINDOW_SIZE]
    if len(frame_ts) < WINDOW_SIZE:
        return None, None, None

    cam0_imgs, cam1_imgs, vis_frames = [], [], []
    for ts in frame_ts:
        img0 = read_gray(cam0_map[ts])
        img1 = read_gray(cam1_map[ts])
        img1_deg = apply_config(img1.copy(), cfg)
        cam0_imgs.append(img0.astype(np.float32) / 255.0)
        cam1_imgs.append(img1_deg.astype(np.float32) / 255.0)
        vis_frames.append((img0, img1_deg))

    cam0 = torch.tensor(np.stack(cam0_imgs)).unsqueeze(1)  # (T,1,H,W)
    cam1 = torch.tensor(np.stack(cam1_imgs)).unsqueeze(1)
    return cam0.unsqueeze(0), cam1.unsqueeze(0), vis_frames  # (1,T,1,H,W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",      default="traj1")
    parser.add_argument("--n-windows", type=int, default=5,
                        help="Windows to sample per condition")
    parser.add_argument("--out",       default=str(ROOT / "analysis" / "model_sanity_check.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load model
    model = StereoHealthMonitor(freeze_backbone=True)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device).eval()
    print(f"Loaded: {CHECKPOINT.name}")

    # load sequence metadata
    seq_dir  = DATASET_ROOT / "sequence" / args.traj
    meta     = json.loads((seq_dir / "meta.json").read_text())
    cam0_dir = Path(meta["cam0_dir"])
    cam1_dir = Path(meta["cam1_dir"])
    cam0_map = {int(f.stem): f for f in sorted(cam0_dir.glob("*.png"))}
    cam1_map = {int(f.stem): f for f in sorted(cam1_dir.glob("*.png"))}
    sorted_ts = sorted(cam0_map.keys())
    print(f"Traj: {args.traj}  |  {len(sorted_ts)} frames")

    # probe each condition
    results = []   # (cond, gt_sev, pred_sev, vis_frames)

    for cond in PROBE_CONDITIONS:
        cond_dir = seq_dir / "degradations" / cond
        if not cond_dir.exists():
            print(f"  [SKIP] {cond} not in dataset index")
            continue

        cfg = json.loads((cond_dir / "config.json").read_text())
        df  = pd.read_csv(cond_dir / "dataset.csv")
        # pick windows with known (non-inf) severity
        valid = df[df["severity"] < 1.0].head(args.n_windows)

        for _, row in valid.iterrows():
            cam0_t, cam1_t, vis = load_window(
                cam0_map, cam1_map, sorted_ts, int(row["window_start_ts"]), cfg)
            if cam0_t is None:
                continue

            with torch.no_grad():
                pred = model(cam0_t.to(device), cam1_t.to(device)).item()

            gt = float(row["severity"])
            results.append((cond, gt, pred, vis))
            print(f"  {cond:<16}  GT={gt:.3f}  Pred={pred:.3f}  "
                  f"{'✓' if abs(pred - gt) < 0.2 else '✗'}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    # One figure per result: 3 rows × 20 cols
    #   Row 0 : filmstrip — left (clean)
    #   Row 1 : filmstrip — right (degraded)
    #   Row 2 : GT vs Pred bar chart (spans full width)
    n = len(results)
    out_path = Path(args.out)

    for row_i, (cond, gt, pred, vis) in enumerate(results):
        fig = plt.figure(figsize=(WINDOW_SIZE * 1.1, 5))
        fig.suptitle(f"{args.traj} / {cond}   |   "
                     f"GT={gt:.3f}   Pred={pred:.3f}   Δ={abs(pred-gt):.3f}   "
                     f"{'✓' if abs(pred-gt) < 0.2 else '✗'}",
                     fontsize=10)

        gs = fig.add_gridspec(3, WINDOW_SIZE, hspace=0.05, wspace=0.02)

        # left filmstrip
        for fi in range(WINDOW_SIZE):
            ax = fig.add_subplot(gs[0, fi])
            ax.imshow(vis[fi][0], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        fig.add_subplot(gs[0, 0]).set_ylabel("left\n(clean)", fontsize=7,
                                              rotation=0, labelpad=40, va="center")

        # right filmstrip
        for fi in range(WINDOW_SIZE):
            ax = fig.add_subplot(gs[1, fi])
            ax.imshow(vis[fi][1], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        fig.add_subplot(gs[1, 0]).set_ylabel("right\n(deg.)", fontsize=7,
                                              rotation=0, labelpad=40, va="center")

        # bar chart
        ax_b = fig.add_subplot(gs[2, :])
        colors = ["#2ca02c" if v < 0.5 else "#d62728" for v in [gt, pred]]
        bars = ax_b.barh(["GT", "Pred"], [gt, pred], color=colors, height=0.5)
        ax_b.axvline(0.5, color="black", linestyle="--", linewidth=1,
                     alpha=0.7, label="crossover (0.5)")
        ax_b.set_xlim(0, 1)
        ax_b.set_xlabel("Severity", fontsize=8)
        ax_b.legend(fontsize=7, loc="upper right")
        for bar, val in zip(bars, [gt, pred]):
            ax_b.text(min(val + 0.01, 0.93), bar.get_y() + bar.get_height() / 2,
                      f"{val:.3f}", va="center", fontsize=8)

        out_i = out_path.parent / f"{out_path.stem}_{row_i:02d}_{cond}.png"
        fig.savefig(out_i, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_i.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    errors = [abs(r[1] - r[2]) for r in results]
    print(f"\nMAE: {np.mean(errors):.4f}  |  Max error: {np.max(errors):.4f}")
    correct = sum(1 for r in results if (r[1] >= 0.5) == (r[2] >= 0.5))
    print(f"Crossover accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")

    # ── Crossover classification summary plot ─────────────────────────────────
    gts   = np.array([r[1] for r in results])
    preds = np.array([r[2] for r in results])
    conds = [r[0] for r in results]

    gt_flag   = gts   >= 0.5   # True = "should switch to mono"
    pred_flag = preds >= 0.5

    tp = np.sum( gt_flag &  pred_flag)   # correctly flagged degraded
    tn = np.sum(~gt_flag & ~pred_flag)   # correctly flagged healthy
    fp = np.sum(~gt_flag &  pred_flag)   # false alarm
    fn = np.sum( gt_flag & ~pred_flag)   # missed degradation

    total    = len(results)
    accuracy = (tp + tn) / total

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Crossover Classification Summary  —  {args.traj}  |  "
        f"Accuracy: {tp+tn}/{total} = {accuracy:.1%}",
        fontsize=11
    )

    # Left: scatter predicted vs ground-truth severity, coloured by correctness
    ax = axes[0]
    correct_mask = gt_flag == pred_flag
    ax.scatter(gts[ correct_mask], preds[ correct_mask],
               color="#2ca02c", alpha=0.7, s=60, label=f"Correct ({correct_mask.sum()})", zorder=3)
    ax.scatter(gts[~correct_mask], preds[~correct_mask],
               color="#d62728", alpha=0.9, s=80, marker="X",
               label=f"Misclassified ({(~correct_mask).sum()})", zorder=4)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="perfect")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.fill_between([0, 0.5], [0.5, 0.5], [1, 1], color="#d62728", alpha=0.04)   # FP zone
    ax.fill_between([0.5, 1], [0,   0  ], [0.5, 0.5], color="#d62728", alpha=0.04)  # FN zone
    ax.set_xlabel("Ground-truth severity", fontsize=10)
    ax.set_ylabel("Predicted severity",   fontsize=10)
    ax.set_title("Predicted vs True Severity\n(coloured by crossover correctness)", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    # annotate quadrants
    for (x, y, txt) in [(0.25, 0.75, "FP\n(false alarm)"),
                         (0.75, 0.25, "FN\n(missed)")]:
        ax.text(x, y, txt, ha="center", va="center", fontsize=7.5,
                color="#d62728", alpha=0.6)
    ax.text(0.25, 0.25, "TN\n(healthy ✓)", ha="center", va="center",
            fontsize=7.5, color="#2ca02c", alpha=0.6)
    ax.text(0.75, 0.75, "TP\n(degraded ✓)", ha="center", va="center",
            fontsize=7.5, color="#2ca02c", alpha=0.6)

    # Right: confusion matrix as heatmap
    ax2 = axes[1]
    cm  = np.array([[tn, fp], [fn, tp]])
    im  = ax2.imshow(cm, cmap="Blues", vmin=0, vmax=total)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=18, fontweight="bold",
                     color="white" if cm[i, j] > total * 0.4 else "black")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred: Healthy\n(< 0.5)", "Pred: Degrade\n(≥ 0.5)"], fontsize=9)
    ax2.set_yticklabels(["GT: Healthy\n(< 0.5)", "GT: Degraded\n(≥ 0.5)"], fontsize=9)
    ax2.set_title(
        f"Confusion Matrix\nTP={tp}  TN={tn}  FP={fp}  FN={fn}  |  Acc={accuracy:.1%}",
        fontsize=9
    )
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    summary_path = out_path.parent / f"{out_path.stem}_crossover_summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {summary_path.name}")


if __name__ == "__main__":
    main()
