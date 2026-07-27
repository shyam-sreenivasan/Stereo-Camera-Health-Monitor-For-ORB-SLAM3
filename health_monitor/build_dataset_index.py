"""
Build dataset.csv and config.json for each degradation condition.

Reads metadata.yaml written by run_pipeline.py to resolve sequence paths,
condition params, and trajectory files. Supports multiple trajectories and
multiple baseline/experiment result folders.

Usage:
    python health_monitor/build_dataset_index.py \
        --baseline clean \
        --experiments new_blur1

    --baseline    : one or more result folder names that contain clean mono
                    trajectories (e.g. 'clean' or 'clean room2_clean')
    --experiments : one or more result folder names with degradation conditions
    --dataset     : optional override for health_monitor_dataset root
    --severity-policy : ratio (default) or log_ratio
    --severity-params : e.g. alpha=3.0 (for log_ratio)
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RESULTS_ROOT  = PROJECT_ROOT / "results"

WINDOW_SIZE   = 20
MAX_DIFF_NS   = 10_000_000   # 10 ms
MIN_COVERAGE  = 0.75


# ── Severity policies ─────────────────────────────────────────────────────────

class SeverityPolicy(ABC):
    @abstractmethod
    def compute(self, rte_s: float, rte_m: float) -> float: ...

    @classmethod
    def from_config(cls, name: str, params: dict) -> "SeverityPolicy":
        if name not in SEVERITY_POLICIES:
            raise ValueError(f"Unknown policy '{name}'. Available: {list(SEVERITY_POLICIES)}")
        return SEVERITY_POLICIES[name](**params)


class RatioPolicy(SeverityPolicy):
    """severity = rte_s / (rte_s + rte_m).  Crossover at rte_s == rte_m → 0.5."""
    def compute(self, rte_s, rte_m):
        denom = rte_s + rte_m
        return float(np.clip(rte_s / denom, 0.0, 1.0)) if denom > 1e-9 else 0.0


class LogRatioPolicy(SeverityPolicy):
    """severity = sigmoid(alpha * log(rte_s / rte_m)).  Scale-agnostic crossover."""
    def __init__(self, alpha: float = 3.0):
        self.alpha = alpha
    def compute(self, rte_s, rte_m):
        if rte_m < 1e-9:
            return 1.0
        return float(1.0 / (1.0 + np.exp(-self.alpha * np.log(rte_s / rte_m))))


SEVERITY_POLICIES = {"ratio": RatioPolicy, "log_ratio": LogRatioPolicy}


# ── Trajectory I/O ────────────────────────────────────────────────────────────

def load_tum_traj(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.array([], dtype=np.int64), np.empty((0, 3))
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return (data[:, 0] * 1e9).astype(np.int64), data[:, 1:4]


def load_gt(path):
    df  = pd.read_csv(path, comment="#", header=None)
    ts  = df.iloc[:, 0].values.astype(np.int64)
    pos = df.iloc[:, 1:4].values.astype(np.float64)
    return ts, pos


def nearest_match(query_ts, ref_ts):
    idx  = np.searchsorted(ref_ts, query_ts)
    idx  = np.clip(idx, 0, len(ref_ts) - 1)
    left = np.clip(idx - 1, 0, len(ref_ts) - 1)
    use_left = np.abs(ref_ts[left] - query_ts) < np.abs(ref_ts[idx] - query_ts)
    idx[use_left] = left[use_left]
    return idx, np.abs(ref_ts[idx] - query_ts) <= MAX_DIFF_NS


def window_rte(pos_est, pos_gt):
    errors = np.linalg.norm(np.diff(pos_est, axis=0) - np.diff(pos_gt, axis=0), axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


# ── Metadata readers ──────────────────────────────────────────────────────────

def read_traj_metadata(result_folder: Path, traj: str) -> dict:
    """Read results/<exp>/<traj>/metadata.yaml."""
    path = result_folder / traj / "metadata.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Traj metadata not found: {path}\n"
            f"Re-run run_pipeline.py to generate metadata."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def read_run_metadata(result_folder: Path, traj: str, cond: str, mode: str) -> dict | None:
    """Read results/<exp>/<traj>/<cond>/<mode>/metadata.yaml. Returns None if missing."""
    path = result_folder / traj / cond / mode / "metadata.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_sequence_paths(sequence_path: str) -> dict:
    """
    Given a sequence_path string (absolute or relative to PROJECT_ROOT),
    return resolved cam0_eq_dir, cam1_eq_dir, gt_csv paths.
    """
    p = Path(sequence_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    return {
        "cam0_eq_dir": p / "mav0" / "cam0" / "data",
        "cam1_eq_dir": p / "mav0" / "cam1" / "data",
        "gt_csv":      p / "mav0" / "mocap0" / "data.csv",
    }


# ── Per-traj data container ───────────────────────────────────────────────────

class TrajData:
    """All precomputed reference data for one trajectory."""

    def __init__(self, traj: str, sequence_path: str):
        self.traj         = traj
        self.sequence_path = sequence_path
        paths = resolve_sequence_paths(sequence_path)

        # camera frames (timestamp → path)
        cam_frames   = sorted(paths["cam0_eq_dir"].glob("*.png"))
        if not cam_frames:
            raise FileNotFoundError(f"No cam0_eq frames found in {paths['cam0_eq_dir']}")
        self.cam_ts  = np.array([int(f.stem) for f in cam_frames], dtype=np.int64)
        self.n_frames = len(self.cam_ts)

        # ground truth
        self.gt_ts, self.gt_pos = load_gt(paths["gt_csv"])

        # paths for meta.json
        self.cam0_eq_dir = str(paths["cam0_eq_dir"])  # stored as cam0_dir in meta.json
        self.cam1_eq_dir = str(paths["cam1_eq_dir"])  # stored as cam1_dir in meta.json

        # mono window RTEs — populated later
        self.mono_window_rtes: dict[int, float | None] = {}

    def load_mono_baseline(self, mono_traj_path: Path):
        mono_ts, mono_pos = load_tum_traj(mono_traj_path)
        if len(mono_ts) == 0:
            raise RuntimeError(f"Mono baseline trajectory is empty: {mono_traj_path}")

        print(f"  [{self.traj}] cam frames={self.n_frames}  gt={len(self.gt_ts)}"
              f"  mono={len(mono_ts)}")

        n_windows = self.n_frames // WINDOW_SIZE
        for w in range(n_windows):
            win_ts = self.cam_ts[w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE]
            g_idx, g_valid = nearest_match(win_ts, self.gt_ts)
            m_idx, m_valid = nearest_match(win_ts, mono_ts)
            both = m_valid & g_valid
            if both.sum() >= int(MIN_COVERAGE * WINDOW_SIZE):
                self.mono_window_rtes[w] = window_rte(
                    mono_pos[m_idx[both]], self.gt_pos[g_idx[both]]
                )
            else:
                self.mono_window_rtes[w] = None


# ── Build baselines ───────────────────────────────────────────────────────────

def build_baseline_map(baseline_names: list[str]) -> dict[str, TrajData]:
    """
    For each baseline result folder, discover trajs via metadata.yaml and
    precompute mono window RTEs.
    Returns {traj_name: TrajData}.
    """
    traj_map: dict[str, TrajData] = {}

    for baseline_name in baseline_names:
        result_folder = RESULTS_ROOT / baseline_name
        if not result_folder.exists():
            raise FileNotFoundError(f"Baseline result folder not found: {result_folder}")

        # discover trajs from subdirectory metadata.yaml files
        for traj_dir in sorted(result_folder.iterdir()):
            if not traj_dir.is_dir():
                continue
            meta_path = traj_dir / "metadata.yaml"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = yaml.safe_load(f)

            traj        = meta["traj"]
            seq_path    = meta["sequence_path"]
            mono_path   = result_folder / traj / "clean" / "mono" / "traj_tum.txt"

            if not mono_path.exists():
                print(f"[WARN] No mono baseline for {baseline_name}/{traj} — skipping")
                continue

            if traj in traj_map:
                print(f"[WARN] Traj '{traj}' already loaded from a previous baseline, skipping")
                continue

            td = TrajData(traj, seq_path)
            td.load_mono_baseline(mono_path)
            traj_map[traj] = td
            print(f"  [{traj}] mono baseline loaded from {baseline_name}")

    if not traj_map:
        raise RuntimeError("No valid baseline trajs found. Check --baseline and results/ folder.")

    return traj_map


# ── Process conditions ────────────────────────────────────────────────────────

def process_experiment(exp_name: str, traj_map: dict[str, TrajData],
                       dataset_root: Path, policy: SeverityPolicy,
                       skip_failed: bool = False):
    result_folder = RESULTS_ROOT / exp_name
    if not result_folder.exists():
        print(f"[WARN] Result folder not found: {result_folder} — skipping")
        return

    print(f"\n=== Experiment: {exp_name} ===")

    for traj_dir in sorted(result_folder.iterdir()):
        if not traj_dir.is_dir():
            continue
        traj_meta_path = traj_dir / "metadata.yaml"
        if not traj_meta_path.exists():
            continue

        with open(traj_meta_path) as f:
            traj_meta = yaml.safe_load(f)
        traj = traj_meta["traj"]

        if traj not in traj_map:
            print(f"[WARN] No baseline found for traj '{traj}' — skipping")
            continue

        td = traj_map[traj]

        # write sequence meta.json (cam dirs for stereo_health_dataset.py)
        seq_out = dataset_root / "sequence" / traj
        seq_out.mkdir(parents=True, exist_ok=True)
        with open(seq_out / "meta.json", "w") as f:
            json.dump({
                "traj":          traj,
                "sequence_path": td.sequence_path,
                "cam0_dir":      td.cam0_eq_dir,
                "cam1_dir":      td.cam1_eq_dir,
            }, f, indent=2)

        # iterate condition dirs
        for cond_dir in sorted(traj_dir.iterdir()):
            if not cond_dir.is_dir() or cond_dir.name == "clean":
                continue

            run_meta = read_run_metadata(result_folder, traj, cond_dir.name, "stereo")
            if run_meta is None:
                print(f"  [SKIP] {traj}/{cond_dir.name} — no run metadata.yaml")
                continue

            cond            = run_meta["condition"]
            condition_params = run_meta.get("condition_params") or {}
            traj_path        = result_folder / traj / cond / "stereo" / "traj_tum.txt"

            print(f"\n  [{traj}/{cond}]")

            out_dir = seq_out / "degradations" / cond
            out_dir.mkdir(parents=True, exist_ok=True)

            # write config.json from run metadata
            with open(out_dir / "config.json", "w") as f:
                json.dump(condition_params if condition_params else {"type": "none"}, f, indent=2)

            # compute windows
            stereo_ts, stereo_pos = load_tum_traj(traj_path)
            n_windows = td.n_frames // WINDOW_SIZE

            if len(stereo_ts) == 0:
                if skip_failed:
                    print(f"    SLAM failed — skipping (--skip-failed)")
                    import shutil
                    shutil.rmtree(out_dir, ignore_errors=True)
                    continue
                print(f"    SLAM failed — all {n_windows} windows → severity=1.0")
                rows = _failed_rows(td.cam_ts, td.mono_window_rtes, n_windows)
            else:
                print(f"    Stereo poses: {len(stereo_ts)}")
                rows = _compute_rows(td, stereo_ts, stereo_pos, n_windows, policy)

                if skip_failed and all(r["severity"] == 1.0 for r in rows):
                    print(f"    All windows severity=1.0 — skipping (--skip-failed)")
                    import shutil
                    shutil.rmtree(out_dir, ignore_errors=True)
                    continue

            df = pd.DataFrame(rows)
            df.to_csv(out_dir / "dataset.csv", index=False)
            sev = df["severity"]
            print(f"    Windows: {len(rows)}  "
                  f"min={sev.min():.3f}  max={sev.max():.3f}  mean={sev.mean():.3f}")


def _failed_rows(cam_ts, mono_window_rtes, n_windows):
    rows = []
    for w in range(n_windows):
        rte_m = mono_window_rtes.get(w)
        rows.append({
            "window_start_ts": int(cam_ts[w * WINDOW_SIZE]),
            "rte_stereo":      float("inf"),
            "rte_mono":        round(rte_m, 6) if rte_m is not None else float("nan"),
            "severity":        1.0,
        })
    return rows


def _compute_rows(td: TrajData, stereo_ts, stereo_pos, n_windows, policy):
    rows = []
    for w in range(n_windows):
        win_ts = td.cam_ts[w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE]
        s_idx, s_valid = nearest_match(win_ts, stereo_ts)
        g_idx, g_valid = nearest_match(win_ts, td.gt_ts)
        both  = s_valid & g_valid
        rte_m = td.mono_window_rtes[w]

        if both.sum() / WINDOW_SIZE < MIN_COVERAGE:
            rows.append({
                "window_start_ts": int(win_ts[0]),
                "rte_stereo":      float("inf"),
                "rte_mono":        round(rte_m, 6) if rte_m is not None else float("nan"),
                "severity":        1.0,
            })
            continue

        rte_s    = window_rte(stereo_pos[s_idx[both]], td.gt_pos[g_idx[both]])
        severity = 1.0 if rte_m is None else policy.compute(rte_s, rte_m)

        rows.append({
            "window_start_ts": int(win_ts[0]),
            "rte_stereo":      round(rte_s, 6),
            "rte_mono":        round(rte_m, 6) if rte_m is not None else float("nan"),
            "severity":        round(severity, 4),
        })
    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True,
                        help="Result folder name(s) containing clean mono trajectories "
                             "(e.g. 'clean' or 'clean room2_clean')")
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="Result folder name(s) with degradation conditions "
                             "(e.g. 'new_blur1 room2_blur')")
    parser.add_argument("--dataset", default=None,
                        help="Path to health_monitor_dataset root "
                             "(default: data/health_monitor_dataset)")
    parser.add_argument("--skip-failed", action="store_true",
                        help="Skip conditions where SLAM fully failed (all windows severity=1.0)")
    parser.add_argument("--severity-policy", default="ratio",
                        choices=list(SEVERITY_POLICIES.keys()))
    parser.add_argument("--severity-params", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    severity_params = {}
    for kv in args.severity_params:
        k, _, v = kv.partition("=")
        try:
            severity_params[k] = float(v)
        except ValueError:
            severity_params[k] = v

    policy = SeverityPolicy.from_config(args.severity_policy, severity_params)
    print(f"Severity policy : {args.severity_policy}  {severity_params or ''}\n")

    dataset_root = Path(args.dataset) if args.dataset \
                   else PROJECT_ROOT / "data" / "health_monitor_dataset"

    print("Loading baselines...")
    traj_map = build_baseline_map(args.baseline)

    for exp_name in args.experiments:
        process_experiment(exp_name, traj_map, dataset_root, policy,
                           skip_failed=args.skip_failed)

    print("\nDone.")
