"""Load, validate, and provide accessors for experiment YAML configs."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from src.config import PROJECT_ROOT, RESULTS_DIR, SEED, TIMEOUT
from src.noise import validate_conditions, build_degradations_from_config


# ---------------------------------------------------------------------------
# ORB-SLAM experiment config (used by run_pipeline.py)
# ---------------------------------------------------------------------------

@dataclass
class OrbslamConfig:
    name: str
    seed: int
    timeout: int
    trajs: dict[str, Path]
    conditions: list[str]
    conditions_cfg: dict        # full params dict {cond: params_or_None}
    slam_modes: list[str]
    results_dir: Path
    dataset_out: Path

    def run_combos(self):
        for traj in self.trajs:
            for cond in self.conditions:
                for mode in self.slam_modes:
                    yield traj, cond, mode

    def run_dir(self, traj: str, condition: str, mode: str) -> Path:
        return self.results_dir / traj / condition / mode

    def euroc_out(self, traj: str) -> Path:
        """Base directory for EuRoC-format noisy datasets for a traj."""
        return self.dataset_out / traj


# ---------------------------------------------------------------------------
# ML experiment config (used by run_ml.py)
# ---------------------------------------------------------------------------

@dataclass
class MLConfig:
    name: str
    seed: int
    trajs: dict[str, Path]
    conditions: list[str]
    dataset_out: Path

    def dataset_dir(self, traj: str, condition: str) -> Path:
        return self.dataset_out / "sequence" / traj / "degradations" / condition


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _resolve_trajs(trajs_raw: dict) -> dict[str, Path]:
    trajs = {}
    for name, path in trajs_raw.items():
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Traj dataset not found: {p}")
        trajs[name] = p
    return trajs


def load_orbslam_config(yaml_path: str | Path) -> OrbslamConfig:
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    name = raw["name"]
    trajs = _resolve_trajs(raw.get("sequences", raw.get("trajs", {})))

    conditions_cfg = raw["conditions"]
    build_degradations_from_config(conditions_cfg)   # register callables from YAML params
    conditions = list(conditions_cfg.keys())         # downstream code still gets a plain list
    validate_conditions(conditions)

    slam_modes = raw.get("slam_modes", ["stereo"])
    valid_modes = {"stereo", "mono"}
    bad = [m for m in slam_modes if m not in valid_modes]
    if bad:
        raise ValueError(f"Invalid slam_modes: {bad}. Choose from {valid_modes}")

    dataset_out_raw = raw.get("dataset_out", f"data/noisy_datasets/{name}")
    dataset_out = Path(dataset_out_raw)
    if not dataset_out.is_absolute():
        dataset_out = PROJECT_ROOT / dataset_out

    return OrbslamConfig(
        name=name,
        seed=raw.get("seed", SEED),
        timeout=raw.get("timeout", TIMEOUT),
        trajs=trajs,
        conditions=conditions,
        conditions_cfg=conditions_cfg,
        slam_modes=slam_modes,
        results_dir=RESULTS_DIR / name,
        dataset_out=dataset_out,
    )


def load_ml_config(yaml_path: str | Path) -> MLConfig:
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    name = raw["name"]
    trajs = _resolve_trajs(raw.get("sequences", raw.get("trajs", {})))

    conditions_cfg = raw["conditions"]
    build_degradations_from_config(conditions_cfg)   # register callables from YAML params
    conditions = list(conditions_cfg.keys())         # downstream code still gets a plain list
    validate_conditions(conditions)

    dataset_out_raw = raw.get("dataset_out", f"data/ml_datasets/{name}")
    dataset_out = Path(dataset_out_raw)
    if not dataset_out.is_absolute():
        dataset_out = PROJECT_ROOT / dataset_out

    return MLConfig(
        name=name,
        seed=raw.get("seed", SEED),
        trajs=trajs,
        conditions=conditions,
        dataset_out=dataset_out,
    )