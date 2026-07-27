"""
StereoHealthDataset — path-based, degradation on-the-fly.

Discovers all trajs and conditions under health_monitor_dataset/sequence/.
For each 20-frame window:
  - cam0: clean frames from cam0_eq_dir (resolved from sequence/<traj>/meta.json)
  - cam1: frames from cam1_eq_dir, degradation applied from config.json
  - severity: precomputed RTE-based score from dataset.csv

Returns:
  cam0     : (T, 1, H, W) float32 tensor  — clean left
  cam1     : (T, 1, H, W) float32 tensor  — degraded right
  severity : scalar float32 tensor
"""

import json
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from config import DATASET_ROOT, WINDOW_SIZE, IMG_SIZE


# ── Degradation functions ─────────────────────────────────────────────────────

def _apply_occlusion(img, frac):
    h, w = img.shape[:2]
    patch_area = int(h * w * frac)
    patch_h    = int(np.sqrt(patch_area * h / w))
    patch_w    = int(patch_area / max(patch_h, 1))
    patch_h, patch_w = min(patch_h, h), min(patch_w, w)
    y = np.random.randint(0, h - patch_h + 1)
    x = np.random.randint(0, w - patch_w + 1)
    out = img.copy()
    out[y:y + patch_h, x:x + patch_w] = 0
    return out


def _apply_config(img, config):
    """Apply degradation from config.json to a grayscale uint8 image."""
    t = config.get("type", "none")

    if t in ("none", "clean"):
        return img

    if t == "gaussian_blur":
        ks = int(config["kernel_size"])
        if ks % 2 == 0:
            ks += 1
        return cv2.GaussianBlur(img, (ks, ks), float(config.get("sigma", 0)))

    if t == "gaussian_noise":
        sigma = float(config["sigma"])
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    if t == "occlusion":
        return _apply_occlusion(img, float(config["frac"]))

    if t == "salt_and_pepper":
        density = float(config["density"])
        out = img.copy()
        n_corrupt = int(img.size * density)
        coords = np.random.choice(img.size, size=n_corrupt, replace=False)
        out.flat[coords[:n_corrupt // 2]] = 255
        out.flat[coords[n_corrupt // 2:]] = 0
        return out

    if t == "motion_blur":
        length = int(config["length"])
        angle  = float(config.get("angle", 0))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        M = cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        return cv2.filter2D(img, -1, kernel)

    if t == "brightness":
        alpha = float(config["alpha"])
        return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown degradation type: {t!r}")


def _load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.resize(img, IMG_SIZE)


# ── Dataset ───────────────────────────────────────────────────────────────────

class StereoHealthDataset(Dataset):
    """
    Discovers all trajs and conditions from health_monitor_dataset/sequence/*/

    Each traj must have:
      sequence/<traj>/meta.json          — cam0_eq_dir, cam1_eq_dir
      sequence/<traj>/degradations/<cond>/config.json
      sequence/<traj>/degradations/<cond>/dataset.csv

    Args:
        dataset_root : Path to health_monitor_dataset/
        conditions   : list of condition names to include (None = all)
        seed         : RNG seed for stochastic degradations
    """

    def __init__(self, dataset_root=DATASET_ROOT, conditions=None, seed=42):
        dataset_root = Path(dataset_root)
        self.rng     = np.random.default_rng(seed)

        # {traj: {ts: Path}} for cam0 and cam1 — populated per traj from meta.json
        self._cam0_by_traj: dict[str, dict[int, Path]] = {}
        self._cam1_by_traj: dict[str, dict[int, Path]] = {}
        self._sorted_ts_by_traj: dict[str, list[int]]  = {}

        # flat list of (start_ts, severity, config, traj)
        self.windows: list[tuple] = []

        seq_root = dataset_root / "sequence"
        if not seq_root.exists():
            raise FileNotFoundError(
                f"No sequence directory found at {seq_root}\n"
                f"Run build_dataset_index.py first."
            )

        for traj_dir in sorted(seq_root.iterdir()):
            if not traj_dir.is_dir():
                continue

            meta_path = traj_dir / "meta.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            traj         = traj_dir.name
            cam0_eq_dir  = Path(meta.get("cam0_dir") or meta["cam0_eq_dir"])
            cam1_eq_dir  = Path(meta.get("cam1_dir") or meta["cam1_eq_dir"])

            cam0_map = {int(f.stem): f for f in sorted(cam0_eq_dir.glob("*.png"))}
            cam1_map = {int(f.stem): f for f in sorted(cam1_eq_dir.glob("*.png"))}

            self._cam0_by_traj[traj]     = cam0_map
            self._cam1_by_traj[traj]     = cam1_map
            self._sorted_ts_by_traj[traj] = sorted(cam0_map.keys())

            degrad_root = traj_dir / "degradations"
            if not degrad_root.exists():
                continue

            for cond_dir in sorted(degrad_root.iterdir()):
                if not cond_dir.is_dir():
                    continue
                if conditions and cond_dir.name not in conditions:
                    continue

                csv_path    = cond_dir / "dataset.csv"
                config_path = cond_dir / "config.json"
                if not csv_path.exists() or not config_path.exists():
                    continue

                with open(config_path) as f:
                    cfg = json.load(f)

                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    self.windows.append((
                        int(row["window_start_ts"]),
                        float(row["severity"]),
                        cfg,
                        traj,
                    ))

        if not self.windows:
            raise RuntimeError(
                f"No windows loaded. Run build_dataset_index.py first.\n"
                f"Looked in: {seq_root}"
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_ts, severity, cfg, traj = self.windows[idx]

        sorted_ts = self._sorted_ts_by_traj[traj]
        cam0_map  = self._cam0_by_traj[traj]
        cam1_map  = self._cam1_by_traj[traj]

        # find start index
        arr       = np.array(sorted_ts)
        start_idx = int(np.argmin(np.abs(arr - start_ts)))
        frame_ts  = sorted_ts[start_idx : start_idx + WINDOW_SIZE]

        cam0_imgs, cam1_imgs = [], []
        for ts in frame_ts:
            img0 = _load_image(cam0_map[ts])
            img1 = _load_image(cam1_map[ts])
            img1 = _apply_config(img1, cfg)
            cam0_imgs.append(img0.astype(np.float32) / 255.0)
            cam1_imgs.append(img1.astype(np.float32) / 255.0)

        cam0 = torch.tensor(np.stack(cam0_imgs)).unsqueeze(1)   # (T, 1, H, W)
        cam1 = torch.tensor(np.stack(cam1_imgs)).unsqueeze(1)

        return cam0, cam1, torch.tensor(severity, dtype=torch.float32)
