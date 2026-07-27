"""
Shared configuration for health monitor scripts.
Import from here instead of redefining in each file.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROJECT_ROOT, SEED

# ── Dataset paths ─────────────────────────────────────────────────────────────
DATASET_ROOT   = PROJECT_ROOT / "data" / "health_monitor_dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── Window / RTE parameters ───────────────────────────────────────────────────
WINDOW_SIZE  = 20            # frames per window — must match build_dataset_index.py
IMG_SIZE     = (224, 224)    # resize target for training images
MAX_DIFF_NS  = 10_000_000    # 10 ms max timestamp mismatch for pose matching
MIN_COVERAGE = 0.75          # min fraction of frames needing a matched pose
