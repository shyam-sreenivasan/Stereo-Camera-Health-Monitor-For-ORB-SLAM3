"""
t-SNE analysis on random stereo image pairs.

Samples 50 stereo pairs from a TUM-VI sequence:
  - 5 clean pairs (no degradation)
  - 45 pairs where the right image is degraded with a randomly chosen
    condition from experiments/*.yaml

For each pair, computes:
  emb(left) - emb(right)  using a frozen ResNet18 backbone

Then runs t-SNE on the 512-d diff vectors and plots coloured by:
  1. Degradation type
  2. Severity level (clean / mild / severe / failed)

Usage:
    python analysis/tsne_pairs.py
    python analysis/tsne_pairs.py --traj data/TUM_original/dataset-room2_512_16
    python analysis/tsne_pairs.py --n-pairs 100 --n-clean 10
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

ROOT = Path(__file__).resolve().parent.parent

# ── Degradation configs to sample from ────────────────────────────────────────
# Maps experiment yaml → list of (condition_name, params_dict, severity_label)
# severity_label: "clean", "mild", "severe", "failed" — assigned by level index
EXPERIMENT_YAMLS = [
    ROOT / "experiments" / "blur_sweep_B.yaml",
    ROOT / "experiments" / "motion_blur_sweep.yaml",
    ROOT / "experiments" / "snp_sweep.yaml",
    ROOT / "experiments" / "brightness_sweep.yaml",
    ROOT / "experiments" / "occlusion_sweep.yaml",
]

# How to bucket conditions by position in the conditions list
def _severity_label(idx, total):
    frac = idx / max(total - 1, 1)
    if frac == 0:
        return "mild"
    elif frac < 0.4:
        return "mild"
    elif frac < 0.75:
        return "severe"
    else:
        return "failed"


def load_all_conditions():
    """Return list of (type_name, params, severity_label) from all experiment yamls."""
    all_conds = []
    for yaml_path in EXPERIMENT_YAMLS:
        if not yaml_path.exists():
            continue
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        conds = list(cfg["conditions"].items())
        for idx, (name, params) in enumerate(conds):
            label = _severity_label(idx, len(conds))
            all_conds.append({
                "name":     name,
                "params":   params,
                "type":     params["type"],
                "severity": label,
            })
    return all_conds


# ── Degradation application ────────────────────────────────────────────────────

def apply_degradation(img, params):
    t = params.get("type", "none")

    if t in ("none", "clean"):
        return img

    if t == "gaussian_blur":
        ks = int(params["kernel_size"])
        if ks % 2 == 0:
            ks += 1
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
        n_corrupt = int(img.size * density)
        coords = np.random.choice(img.size, size=n_corrupt, replace=False)
        out.flat[coords[:n_corrupt // 2]] = 255
        out.flat[coords[n_corrupt // 2:]] = 0
        return out

    if t == "brightness":
        alpha = float(params["alpha"])
        return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    if t == "occlusion":
        frac = float(params["frac"])
        h, w = img.shape[:2]
        area = int(h * w * frac)
        ph   = int(np.sqrt(area * h / w))
        pw   = int(area / max(ph, 1))
        ph, pw = min(ph, h), min(pw, w)
        y = np.random.randint(0, h - ph + 1)
        x = np.random.randint(0, w - pw + 1)
        out = img.copy()
        out[y:y + ph, x:x + pw] = 0
        return out

    raise ValueError(f"Unknown degradation type: {t!r}")


# ── Image loading ──────────────────────────────────────────────────────────────

def load_frames(traj_dir: Path):
    cam0_dir = traj_dir / "mav0" / "cam0" / "data"
    cam1_dir = traj_dir / "mav0" / "cam1" / "data"
    frames0  = sorted(cam0_dir.glob("*.png"))
    frames1  = sorted(cam1_dir.glob("*.png"))
    # align by stem (timestamp)
    stems0 = {f.stem: f for f in frames0}
    stems1 = {f.stem: f for f in frames1}
    common = sorted(set(stems0) & set(stems1))
    return [(stems0[s], stems1[s]) for s in common]


def read_gray(path, size=(224, 224)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.resize(img, size)


# ── ResNet18 diff embedding extractor ─────────────────────────────────────────

class DiffExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 1, H, W) → (B, 512)"""
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x).flatten(1)

    @torch.no_grad()
    def embed_pair(self, left_np, right_np, device, mode="concat"):
        """
        left_np, right_np: (H, W) uint8 grayscale
        mode="concat" → 1024-d [left_emb, right_emb]
        mode="diff"   → 512-d  (left_emb - right_emb)
        """
        def to_tensor(img):
            t = torch.tensor(img.astype(np.float32) / 255.0)
            return t.unsqueeze(0).unsqueeze(0).to(device)

        e0 = self(to_tensor(left_np))    # (1, 512)
        e1 = self(to_tensor(right_np))   # (1, 512)
        if mode == "diff":
            return (e0 - e1).squeeze(0).cpu().numpy()                   # (512,)
        return torch.cat([e0, e1], dim=1).squeeze(0).cpu().numpy()      # (1024,)


# ── Sample pairs ──────────────────────────────────────────────────────────────

def sample_pairs(all_frames, all_conditions, n_pairs, n_clean, seed):
    rng = np.random.default_rng(seed)
    chosen_frames = rng.choice(len(all_frames), size=n_pairs, replace=False)

    pairs = []

    # clean pairs
    for i in chosen_frames[:n_clean]:
        p0, p1 = all_frames[i]
        pairs.append({
            "left_path":  p0,
            "right_path": p1,
            "params":     {"type": "clean"},
            "type":       "clean",
            "severity":   "clean",
            "cond_name":  "clean",
        })

    # degraded pairs
    for i in chosen_frames[n_clean:]:
        p0, p1 = all_frames[i]
        cond = all_conditions[rng.integers(len(all_conditions))]
        pairs.append({
            "left_path":  p0,
            "right_path": p1,
            "params":     cond["params"],
            "type":       cond["type"],
            "severity":   cond["severity"],
            "cond_name":  cond["name"],
        })

    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",     default=str(ROOT / "data/TUM_original/dataset-room1_512_16"),
                        help="Path to TUM-VI sequence directory")
    parser.add_argument("--n-pairs",  type=int, default=50)
    parser.add_argument("--n-clean",  type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--mode",     default="concat", choices=["concat", "diff"],
                        help="Embedding mode: concat=[left,right] (1024-d), diff=left-right (512-d)")
    parser.add_argument("--out",      default=None,
                        help="Output PNG path (default: analysis/tsne_pairs_<mode>.png)")
    args = parser.parse_args()

    traj_dir   = Path(args.traj)
    out_path   = Path(args.out) if args.out else ROOT / "analysis" / f"tsne_pairs_{args.mode}.png"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Traj   : {traj_dir}")

    # load frame index
    all_frames = load_frames(traj_dir)
    print(f"Frames : {len(all_frames)} stereo pairs available")

    # load all degradation conditions
    all_conditions = load_all_conditions()
    print(f"Conditions : {len(all_conditions)} loaded from experiment YAMLs")

    # sample pairs
    pairs = sample_pairs(all_frames, all_conditions, args.n_pairs, args.n_clean, args.seed)
    print(f"Sampled: {args.n_pairs} pairs ({args.n_clean} clean, {args.n_pairs - args.n_clean} degraded)")

    # extract diff embeddings
    extractor = DiffExtractor().to(device).eval()
    print("Extracting ResNet18 diff embeddings...")

    embeddings  = []
    types       = []
    severities  = []
    cond_names  = []

    for i, pair in enumerate(pairs):
        left  = read_gray(pair["left_path"])
        right = read_gray(pair["right_path"])
        right_deg = apply_degradation(right.copy(), pair["params"])

        emb = extractor.embed_pair(left, right_deg, device, mode=args.mode)
        embeddings.append(emb)
        types.append(pair["type"])
        severities.append(pair["severity"])
        cond_names.append(pair["cond_name"])

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_pairs} done")

    embeddings = np.stack(embeddings)   # (N, 512)
    print(f"Embeddings shape: {embeddings.shape}")

    # t-SNE
    print("Running t-SNE...")
    tsne   = TSNE(n_components=2, perplexity=min(30, args.n_pairs // 3),
                  random_state=args.seed, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    # ── Plotting ──────────────────────────────────────────────────────────────
    TYPE_COLORS = {
        "clean":          "#2ca02c",
        "gaussian_blur":  "#1f77b4",
        "motion_blur":    "#17becf",
        "salt_and_pepper":"#ff7f0e",
        "brightness":     "#9467bd",
        "occlusion":      "#d62728",
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
    title = ("t-SNE of ResNet18 Diff Embeddings  (left − right)"
             if args.mode == "diff" else
             "t-SNE of ResNet18 Pair Embeddings  [left_emb, right_emb]")
    fig.suptitle(title, fontsize=14)

    # Panel 1 — colour by degradation type
    ax = axes[0]
    for t in TYPE_ORDER:
        mask = np.array([x == t for x in types])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=TYPE_COLORS.get(t, "grey"),
                       label=t.replace("_", " "), s=60, alpha=0.85,
                       edgecolors="white", linewidths=0.4)
    ax.set_title("Coloured by degradation type")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)

    # Panel 2 — colour by severity level
    ax = axes[1]
    for s in SEV_ORDER:
        mask = np.array([x == s for x in severities])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=SEV_COLORS[s],
                       label=s, s=60, alpha=0.85,
                       edgecolors="white", linewidths=0.4)
    ax.set_title("Coloured by severity level")
    ax.set_xlabel("t-SNE dim 1")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    # print pair summary
    print(f"\n{'Type':<20} {'Severity':<10} Count")
    print("─" * 40)
    from collections import Counter
    for (t, s), count in sorted(Counter(zip(types, severities)).items()):
        print(f"  {t:<18} {s:<10} {count}")


if __name__ == "__main__":
    main()
