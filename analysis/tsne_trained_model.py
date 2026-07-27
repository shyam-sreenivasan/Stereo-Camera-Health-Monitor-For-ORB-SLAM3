"""
t-SNE analysis using the trained StereoHealthMonitor's GRU hidden state.

Loads checkpoints/best_model.pt and extracts the 64-d GRU hidden state
for 20-frame windows — the exact representation the classifier acts on.

This is the correct test of whether the trained model has learned
severity-aware representations.

Usage:
    python analysis/tsne_trained_model.py
    python analysis/tsne_trained_model.py --n-windows 300 --n-clean 30
    python analysis/tsne_trained_model.py --traj data/TUM_original/dataset-room2_512_16
"""

import argparse
import sys
import cv2
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "health_monitor"))

CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"

EXPERIMENT_YAMLS = [
    ROOT / "experiments" / "blur_sweep_B.yaml",
    ROOT / "experiments" / "motion_blur_sweep.yaml",
    ROOT / "experiments" / "snp_sweep.yaml",
    ROOT / "experiments" / "brightness_sweep.yaml",
    ROOT / "experiments" / "occlusion_sweep.yaml",
]


# ── Model (must match train.py exactly) ───────────────────────────────────────

class FrameEncoder(nn.Module):
    def __init__(self, embed_dim=128, freeze_backbone=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.project = nn.Sequential(
            nn.Linear(1536, embed_dim),
            nn.ReLU(),
        )

    def encode_image(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x).flatten(1)

    def forward(self, cam0, cam1):
        left_emb  = self.encode_image(cam0)
        right_emb = self.encode_image(cam1)
        diff_emb  = left_emb - right_emb
        return self.project(torch.cat([left_emb, right_emb, diff_emb], dim=1))


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
        cam0_flat  = cam0.view(B * T, C, H, W)
        cam1_flat  = cam1.view(B * T, C, H, W)
        frame_embs = self.encoder(cam0_flat, cam1_flat).view(B, T, -1)
        _, hidden  = self.gru(frame_embs)
        return self.classifier(hidden.squeeze(0)).squeeze(1)


# ── Degradation conditions ─────────────────────────────────────────────────────

def _severity_label(idx, total):
    frac = idx / max(total - 1, 1)
    if frac == 0:       return "mild"
    elif frac < 0.4:    return "mild"
    elif frac < 0.75:   return "severe"
    else:               return "failed"


def load_all_conditions():
    all_conds = []
    for yaml_path in EXPERIMENT_YAMLS:
        if not yaml_path.exists():
            continue
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        conds = list(cfg["conditions"].items())
        for idx, (name, params) in enumerate(conds):
            all_conds.append({
                "name":     name,
                "params":   params,
                "type":     params["type"],
                "severity": _severity_label(idx, len(conds)),
            })
    return all_conds


# ── Degradation application ────────────────────────────────────────────────────

def apply_degradation(img, params):
    t = params.get("type", "none")
    if t in ("none", "clean"):
        return img
    if t == "gaussian_blur":
        ks = int(params["kernel_size"])
        if ks % 2 == 0: ks += 1
        return cv2.GaussianBlur(img, (ks, ks), float(params.get("sigma", 0)))
    if t == "motion_blur":
        length = int(params["length"])
        angle  = float(params.get("angle", 0))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        M = cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        return cv2.filter2D(img, -1, kernel)
    if t == "salt_and_pepper":
        density = float(params["density"])
        out = img.copy()
        n = int(img.size * density)
        coords = np.random.choice(img.size, size=n, replace=False)
        out.flat[coords[:n // 2]] = 255
        out.flat[coords[n // 2:]] = 0
        return out
    if t == "brightness":
        return np.clip(img.astype(np.float32) * float(params["alpha"]), 0, 255).astype(np.uint8)
    if t == "occlusion":
        frac = float(params["frac"])
        h, w = img.shape[:2]
        area = int(h * w * frac)
        ph = int(np.sqrt(area * h / w)); pw = int(area / max(ph, 1))
        ph, pw = min(ph, h), min(pw, w)
        y = np.random.randint(0, h - ph + 1)
        x = np.random.randint(0, w - pw + 1)
        out = img.copy(); out[y:y+ph, x:x+pw] = 0
        return out
    raise ValueError(f"Unknown degradation type: {t!r}")


# ── Frame loading ──────────────────────────────────────────────────────────────

WINDOW_SIZE = 20


def load_frames(traj_dir):
    cam0_dir = traj_dir / "mav0" / "cam0" / "data"
    cam1_dir = traj_dir / "mav0" / "cam1" / "data"
    stems0 = {f.stem: f for f in sorted(cam0_dir.glob("*.png"))}
    stems1 = {f.stem: f for f in sorted(cam1_dir.glob("*.png"))}
    common = sorted(set(stems0) & set(stems1))
    return [(stems0[s], stems1[s]) for s in common]


def read_gray(path, size=(224, 224)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.resize(img, size)


def window_to_tensor(frame_paths, params, device):
    """
    Load a 20-frame window, degrade the right image, return
    cam0: (1, T, 1, H, W), cam1: (1, T, 1, H, W) tensors.
    """
    cam0_imgs, cam1_imgs = [], []
    for p0, p1 in frame_paths:
        left  = read_gray(p0)
        right = apply_degradation(read_gray(p1), params)
        cam0_imgs.append(left.astype(np.float32)  / 255.0)
        cam1_imgs.append(right.astype(np.float32) / 255.0)

    cam0 = torch.tensor(np.stack(cam0_imgs)).unsqueeze(1)   # (T, 1, H, W)
    cam1 = torch.tensor(np.stack(cam1_imgs)).unsqueeze(1)
    return cam0.unsqueeze(0).to(device), cam1.unsqueeze(0).to(device)  # (1, T, 1, H, W)


# ── GRU hidden state extraction ───────────────────────────────────────────────

@torch.no_grad()
def extract_gru_hidden(model, cam0, cam1):
    """
    Run the full model up to (but not including) the classifier.
    Returns the 64-d GRU hidden state.
    """
    B, T, C, H, W = cam0.shape
    cam0_flat  = cam0.view(B * T, C, H, W)
    cam1_flat  = cam1.view(B * T, C, H, W)
    frame_embs = model.encoder(cam0_flat, cam1_flat).view(B, T, -1)
    _, hidden  = model.gru(frame_embs)
    return hidden.squeeze(0).squeeze(0).cpu().numpy()   # (64,)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",      default=str(ROOT / "data/TUM_original/dataset-room1_512_16"))
    parser.add_argument("--n-windows", type=int, default=300)
    parser.add_argument("--n-clean",   type=int, default=30)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       default=str(ROOT / "analysis" / "tsne_trained_model.png"))
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traj_dir = Path(args.traj)
    print(f"Device     : {device}")
    print(f"Traj       : {traj_dir}")
    print(f"Checkpoint : {CHECKPOINT}")

    # load model
    model = StereoHealthMonitor(freeze_backbone=True)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device).eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # load frames + conditions
    all_frames     = load_frames(traj_dir)
    all_conditions = load_all_conditions()
    # usable window start indices (need WINDOW_SIZE consecutive frames)
    n_windows_avail = len(all_frames) - WINDOW_SIZE + 1
    print(f"Frames     : {len(all_frames)} stereo pairs → {n_windows_avail} windows available")
    print(f"Conditions : {len(all_conditions)} from experiment YAMLs")

    # sample window start indices
    rng     = np.random.default_rng(args.seed)
    starts  = rng.choice(n_windows_avail, size=args.n_windows, replace=False)

    windows = []
    for i in starts[:args.n_clean]:
        windows.append({"frames": all_frames[i:i+WINDOW_SIZE],
                        "params": {"type": "clean"},
                        "type": "clean", "severity": "clean"})
    for i in starts[args.n_clean:]:
        cond = all_conditions[rng.integers(len(all_conditions))]
        windows.append({"frames": all_frames[i:i+WINDOW_SIZE],
                        "params": cond["params"],
                        "type": cond["type"], "severity": cond["severity"]})

    # extract 64-d GRU hidden states
    print(f"\nExtracting 64-d GRU hidden states for {args.n_windows} windows...")
    embeddings = []
    types      = []
    severities = []

    for i, win in enumerate(windows):
        cam0, cam1 = window_to_tensor(win["frames"], win["params"], device)
        emb        = extract_gru_hidden(model, cam0, cam1)
        embeddings.append(emb)
        types.append(win["type"])
        severities.append(win["severity"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.n_windows} done")

    embeddings = np.stack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # t-SNE
    print("Running t-SNE...")
    tsne   = TSNE(n_components=2, perplexity=40, random_state=args.seed, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    # ── Plot ──────────────────────────────────────────────────────────────────
    TYPE_COLORS = {

        "clean":           "#2ca02c",
        "gaussian_blur":   "#1f77b4",
        "motion_blur":     "#17becf",
        "salt_and_pepper": "#ff7f0e",
        "brightness":      "#9467bd",
        "occlusion":       "#d62728",
    }
    SEV_COLORS = {
        "clean":  "#2ca02c",
        "mild":   "#aec7e8",
        "severe": "#ff7f0e",
        "failed": "#d62728",
    }
    SEV_ORDER  = ["clean", "mild", "severe", "failed"]
    TYPE_ORDER = ["clean", "gaussian_blur", "motion_blur",
                  "salt_and_pepper", "brightness", "occlusion"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("t-SNE of Trained StereoHealthMonitor — GRU Hidden State (64-d)\n"
                 "20-frame windows, full model forward pass", fontsize=13)

    ax = axes[0]
    for t in TYPE_ORDER:
        mask = np.array([x == t for x in types])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=TYPE_COLORS.get(t, "grey"),
                       label=t.replace("_", " "), s=25, alpha=0.8,
                       edgecolors="white", linewidths=0.3)
    ax.set_title("Coloured by degradation type")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    ax = axes[1]
    for s in SEV_ORDER:
        mask = np.array([x == s for x in severities])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=SEV_COLORS[s],
                       label=s, s=25, alpha=0.8,
                       edgecolors="white", linewidths=0.3)
    ax.set_title("Coloured by severity level")
    ax.set_xlabel("t-SNE dim 1")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")

    print(f"\n{'Type':<20} {'Severity':<10} Count")
    print("─" * 40)
    for (t, s), count in sorted(Counter(zip(types, severities)).items()):
        print(f"  {t:<18} {s:<10} {count}")



if __name__ == "__main__":
    main()
