"""
t-SNE analysis of diff embeddings.

For each window we compute:
  mean_diff = mean over 10 frames of (left_emb - right_emb)

Then run t-SNE on these vectors and plot coloured by:
  1. ks value       — do blur levels separate?
  2. severity       — is there a smooth gradient?
  3. healthy/degraded — is there a clean binary split?
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from torch.utils.data import Dataset


class StereoHealthDataset(Dataset):
    """
    Each item:
      cam0     : (10, 1, H, W) tensor — clean left frames
      cam1     : (10, 1, H, W) tensor — blurred right frames
      severity : scalar tensor in [0, 1]
    """
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        cam0 = torch.tensor(w["cam0"]).unsqueeze(1)        # (10, 1, H, W)
        cam1 = torch.tensor(w["cam1"]).unsqueeze(1)        # (10, 1, H, W)
        severity = torch.tensor(w["severity"], dtype=torch.float32)
        return cam0, cam1, severity

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH   = Path("datasets/health_monitor_dataset.pkl")
CHECKPOINT_DIR = Path("checkpoints")
N_SAMPLES      = 500    # subsample for speed — t-SNE is O(n^2)
SEED           = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Backbone only (no GRU, no classifier) ────────────────────────────────────
class DiffEmbeddingExtractor(nn.Module):
    """
    Extracts the mean diff embedding over a 10-frame window.
    Uses the same shared ResNet18 backbone as the full model.
    Does NOT include the projection layer, GRU, or classifier —
    we want the raw 512-d diff to see what ResNet captures directly.
    """
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

    def encode(self, x):
        """x: (B, 1, H, W) → (B, 512)"""
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x).flatten(1)

    @torch.no_grad()
    def forward(self, cam0, cam1):
        """
        cam0, cam1: (B, T, 1, H, W)
        returns mean_diff: (B, 512)
        """
        B, T, C, H, W = cam0.shape
        cam0_flat = cam0.view(B * T, C, H, W)
        cam1_flat = cam1.view(B * T, C, H, W)

        left_emb  = self.encode(cam0_flat)              # (B*T, 512)
        right_emb = self.encode(cam1_flat)              # (B*T, 512)
        diff      = left_emb - right_emb                # (B*T, 512)

        # mean diff over the 10 frames in each window
        diff = diff.view(B, T, 512)                     # (B, T, 512)
        return diff.mean(dim=1)                         # (B, 512)


# ── Extract embeddings ────────────────────────────────────────────────────────
def extract_embeddings(extractor, windows, n_samples, device):
    """
    Subsample windows, extract diff embeddings, return
    embeddings + metadata (severity, ks).
    """
    # subsample
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(windows), size=min(n_samples, len(windows)),
                     replace=False)
    sampled = [windows[i] for i in idx]

    dataset = StereoHealthDataset(sampled)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_embs       = []
    all_severities = []
    all_ks         = []

    extractor.eval()
    for cam0, cam1, severity in loader:
        cam0 = cam0.to(device)
        cam1 = cam1.to(device)

        emb = extractor(cam0, cam1)           # (B, 512)
        all_embs.append(emb.cpu().numpy())
        all_severities.extend(severity.numpy())
        all_ks.extend([w["ks"] for w in
                        sampled[len(all_severities)-len(severity):
                                len(all_severities)]])

    embs       = np.vstack(all_embs)
    severities = np.array(all_severities)
    ks_vals    = np.array([w["ks"] for w in sampled])

    return embs, severities, ks_vals


# ── t-SNE ─────────────────────────────────────────────────────────────────────
def run_tsne(embs):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED,
                max_iter=1000, verbose=1)
    return tsne.fit_transform(embs)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_tsne(coords, severities, ks_vals):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("t-SNE of Mean Diff Embeddings (left_emb - right_emb)",
                 fontsize=13)

    # ── Plot 1: coloured by severity (continuous) ────────────────────────────
    ax = axes[0]
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=severities, cmap="RdYlGn_r",
                    s=15, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="severity")
    ax.set_title("Coloured by severity")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)

    # ── Plot 2: coloured by ks level (discrete) ──────────────────────────────
    ax = axes[1]
    unique_ks = sorted(set(ks_vals))
    cmap      = plt.cm.get_cmap("tab20", len(unique_ks))
    for i, ks in enumerate(unique_ks):
        mask = ks_vals == ks
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=cmap(i), label=f"ks={ks}",
                   s=15, alpha=0.7)
    ax.set_title("Coloured by blur level (ks)")
    ax.set_xlabel("t-SNE dim 1")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    # ── Plot 3: coloured by healthy/degraded (binary) ────────────────────────
    ax = axes[2]
    healthy_mask  = severities <  0.5
    degraded_mask = severities >= 0.5
    ax.scatter(coords[healthy_mask,  0], coords[healthy_mask,  1],
               color="steelblue", label="healthy  (<0.5)",
               s=15, alpha=0.7)
    ax.scatter(coords[degraded_mask, 0], coords[degraded_mask, 1],
               color="crimson",   label="degraded (≥0.5)",
               s=15, alpha=0.7)
    ax.set_title("Coloured by crossover flag")
    ax.set_xlabel("t-SNE dim 1")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("tsne_diff_embeddings.png", dpi=150, bbox_inches="tight")
    print("Saved tsne_diff_embeddings.png")
    plt.close()

    # ── Also plot diff embedding norm distribution ────────────────────────────
    # this is a 1D sanity check: does ||diff|| grow with ks?
    fig, ax = plt.subplots(figsize=(8, 4))
    unique_ks = sorted(set(ks_vals))
    means, stds = [], []
    for ks in unique_ks:
        # we don't have the raw embs here so use severity as proxy
        mask = ks_vals == ks
        means.append(severities[mask].mean())
        stds.append(severities[mask].std())

    ax.bar(unique_ks, means, yerr=stds, capsize=3,
           color=["steelblue" if s < 0.5 else "crimson" for s in means],
           alpha=0.8)
    ax.axhline(0.5, color="black", linestyle="--", label="crossover threshold")
    ax.set_xlabel("Blur level (ks)")
    ax.set_ylabel("Mean severity")
    ax.set_title("Mean severity per blur level")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("severity_per_ks.png", dpi=150)
    print("Saved severity_per_ks.png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    with open(DATASET_PATH, "rb") as f:
        windows = pickle.load(f)
    print(f"Dataset: {len(windows)} windows")

    extractor = DiffEmbeddingExtractor().to(DEVICE)
    print(f"Extractor params: "
          f"{sum(p.numel() for p in extractor.parameters()):,} "
          f"(all frozen)")

    embs, severities, ks_vals = extract_embeddings(
        extractor, windows, N_SAMPLES, DEVICE)
    print(f"Embeddings: {embs.shape}  "
          f"severity range: [{severities.min():.3f}, {severities.max():.3f}]")

    coords = run_tsne(embs)
    plot_tsne(coords, severities, ks_vals)

    print("\nDone. Check:")
    print("  tsne_diff_embeddings.png  — 3-panel t-SNE")
    print("  severity_per_ks.png       — mean severity per blur level")


if __name__ == "__main__":
    main()