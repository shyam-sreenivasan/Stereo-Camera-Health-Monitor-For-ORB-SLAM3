import cv2
import shutil
import yaml
import numpy as np
from pathlib import Path
from src.config import ORIG_DATASET, NOISY_BASE, SEED

# ---------------------------------------------------------------------------
# Output root for ML dataset
# ---------------------------------------------------------------------------
ML_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "ml_datasets"

# ---------------------------------------------------------------------------
# Degradation primitives
# ---------------------------------------------------------------------------

def add_gaussian_noise(img, sigma, rng):
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def apply_blur(img, kernel_size, sigma=0):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_motion_blur(img, length, angle=0):
    """Apply directional motion blur with a linear kernel of given length and angle (degrees)."""
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    center = (length // 2, length // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    return cv2.filter2D(img, -1, kernel)


def apply_brightness(img, alpha):
    """Scale pixel intensity by alpha. Values clipped to [0, 255]."""
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def apply_salt_and_pepper(img, density, rng):
    """Corrupt `density` fraction of pixels to 0 (pepper) or 255 (salt), split evenly."""
    out = img.copy()
    n_pixels = img.size
    n_corrupt = int(n_pixels * density)
    coords = rng.choice(n_pixels, size=n_corrupt, replace=False)
    salt_coords = coords[:n_corrupt // 2]
    pepper_coords = coords[n_corrupt // 2:]
    out.flat[salt_coords] = 255
    out.flat[pepper_coords] = 0
    return out


def apply_occlusion(img, frac, rng):
    """Black-out a random rectangular patch covering `frac` of image area."""
    h, w = img.shape[:2]
    patch_area = int(h * w * frac)
    patch_h = int(np.sqrt(patch_area * h / w))
    patch_w = int(patch_area / max(patch_h, 1))
    patch_h, patch_w = min(patch_h, h), min(patch_w, w)
    y = rng.integers(0, h - patch_h + 1)
    x = rng.integers(0, w - patch_w + 1)
    out = img.copy()
    out[y:y + patch_h, x:x + patch_w] = 0
    return out


DEGRADATIONS = {}

# ---------------------------------------------------------------------------
# Config-driven registry builder
# ---------------------------------------------------------------------------

# Supported type strings and their builder functions.
# Each builder receives the raw param dict from YAML and returns a
# (img, rng) -> img callable.

_TYPE_BUILDERS: dict[str, callable] = {}


def _register_type(type_name: str):
    def decorator(fn):
        _TYPE_BUILDERS[type_name] = fn
        return fn
    return decorator


@_register_type("gaussian_blur")
def _build_gaussian_blur(params: dict):
    kernel_size = int(params["kernel_size"])
    sigma = float(params.get("sigma", 0))
    if kernel_size % 2 == 0:
        raise ValueError(
            f"gaussian_blur kernel_size must be odd, got {kernel_size}"
        )
    return lambda img, rng: apply_blur(img, kernel_size, sigma)


@_register_type("gaussian_noise")
def _build_gaussian_noise(params: dict):
    sigma = float(params["sigma"])
    return lambda img, rng: add_gaussian_noise(img, sigma, rng)


@_register_type("motion_blur")
def _build_motion_blur(params: dict):
    length = int(params["length"])
    angle  = float(params.get("angle", 0))
    return lambda img, rng: apply_motion_blur(img, length, angle)


@_register_type("brightness")
def _build_brightness(params: dict):
    alpha = float(params["alpha"])
    return lambda img, rng: apply_brightness(img, alpha)


@_register_type("salt_and_pepper")
def _build_salt_and_pepper(params: dict):
    density = float(params["density"])
    return lambda img, rng: apply_salt_and_pepper(img, density, rng)


@_register_type("occlusion")
def _build_occlusion(params: dict):
    frac = float(params["frac"])
    return lambda img, rng: apply_occlusion(img, frac, rng)


def build_degradations_from_config(conditions_cfg: dict) -> None:

    for key, params in conditions_cfg.items():
        if params is None or key == "clean":
            DEGRADATIONS[key] = None
            continue

        deg_type = params.get("type")
        if deg_type is None:
            raise ValueError(
                f"Condition '{key}' is missing a 'type' field. "
                f"Available types: {list(_TYPE_BUILDERS)}"
            )
        if deg_type not in _TYPE_BUILDERS:
            raise ValueError(
                f"Unknown degradation type '{deg_type}' for condition '{key}'. "
                f"Available types: {list(_TYPE_BUILDERS)}"
            )

        DEGRADATIONS[key] = _TYPE_BUILDERS[deg_type](params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_degradations() -> list[str]:
    """Return all registered degradation keys (for CLI help / validation)."""
    return list(DEGRADATIONS.keys())


def validate_conditions(conditions: list[str]):
    """Raise ValueError if any condition is not in the registry."""
    unknown = [c for c in conditions if c not in DEGRADATIONS]
    if unknown:
        raise ValueError(
            f"Unknown conditions: {unknown}\n"
            f"Available: {list_degradations()}"
        )


# ---------------------------------------------------------------------------
# EuRoC-format dataset generation (used by run_pipeline.py / ORB-SLAM)
# ---------------------------------------------------------------------------

def generate_noisy_dataset_euroc(
    condition: str,
    orig_dataset: Path,
    out_base: Path,
    seed: int = SEED,
):
    """Generate a noisy EuRoC-layout dataset for a single condition.

    Produces out_base/{condition}/mav0/cam0, cam1, imu0, mocap0
    with cam0/imu0/mocap0 symlinked and cam1 degraded.
    Returns the dataset root path.
    """
    validate_conditions([condition])
    degrade_fn = DEGRADATIONS[condition]

    if degrade_fn is None:
        return orig_dataset

    out_dir = out_base / condition
    mav_out = out_dir / "mav0"

    cam1_in_data = orig_dataset / "mav0" / "cam1" / "data"
    cam1_out_data = mav_out / "cam1" / "data"

    # Skip check
    if cam1_out_data.exists():
        n_existing = len(list(cam1_out_data.glob("*.png")))
        n_original = len(list(cam1_in_data.glob("*.png")))
        if n_existing == n_original:
            print(f"  [{condition}] Already generated, skipping.")
            return out_dir

    print(f"  [{condition}] Generating EuRoC dataset...")

    # Copy cam0
    cam0_out = mav_out / "cam0"
    cam0_out.mkdir(parents=True, exist_ok=True)
    cam0_data_dst = cam0_out / "data"
    if not cam0_data_dst.exists():
        cam0_src = orig_dataset / "mav0" / "cam0" / "data"
        _copy_images(cam0_src, cam0_data_dst)
    cam0_csv_src = orig_dataset / "mav0" / "cam0" / "data.csv"
    cam0_csv_dst = cam0_out / "data.csv"
    if cam0_csv_src.exists() and not cam0_csv_dst.exists():
        shutil.copy2(cam0_csv_src, cam0_csv_dst)

    # Copy imu0 and mocap0
    for folder in ["imu0", "mocap0"]:
        dst = mav_out / folder
        src = orig_dataset / "mav0" / folder
        if not dst.exists() and src.exists():
            shutil.copytree(src, dst)

    # Copy cam1 csv
    cam1_out_data.mkdir(parents=True, exist_ok=True)
    cam1_csv_src = orig_dataset / "mav0" / "cam1" / "data.csv"
    cam1_csv_dst = mav_out / "cam1" / "data.csv"
    if cam1_csv_src.exists() and not cam1_csv_dst.exists():
        shutil.copy2(cam1_csv_src, cam1_csv_dst)

    # Apply degradation
    rng = np.random.default_rng(seed)
    images = sorted(cam1_in_data.glob("*.png"))
    for i, img_path in enumerate(images):
        out_path = cam1_out_data / img_path.name
        if out_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        noisy = degrade_fn(img, rng)
        cv2.imwrite(str(out_path), noisy)
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(images)}]")

    return out_dir


# ---------------------------------------------------------------------------
# ML dataset generation
# ---------------------------------------------------------------------------

def _copy_images(src_dir: Path, dst_dir: Path):
    """Copy all PNGs from src to dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(src_dir.glob("*.png")):
        shutil.copy2(img_path, dst_dir / img_path.name)


def _degrade_images(src_dir: Path, dst_dir: Path, degrade_fn, rng):
    """Apply degrade_fn to every PNG in src_dir, write results to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(src_dir.glob("*.png"))
    for i, img_path in enumerate(images):
        out_path = dst_dir / img_path.name
        if out_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        degraded = degrade_fn(img, rng)
        cv2.imwrite(str(out_path), degraded)
        if (i + 1) % 200 == 0:
            print(f"      [{i+1}/{len(images)}]")


def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def generate_ml_dataset(
    trajs: dict[str, Path],
    conditions: list[str],
    out_root: Path | None = None,
):
    """Generate the full ML-ready dataset.

    Args:
        trajs: mapping traj name -> path to EuRoC-format dataset root
               e.g. {"traj1": Path("datasets/V1_01_easy")}
        conditions: list of keys into DEGRADATIONS, e.g.
                    ["clean", "blur_ks3", "blur_ks7", "gauss_4sig"]
        out_root: override output directory (default: top/data/)
    """
    out_root = out_root or ML_DATA_ROOT
    rng = np.random.default_rng(SEED)
    index_entries = {}

    # ------------------------------------------------------------------
    # Shared mocap ground truth  (take from first traj, or merge later)
    # ------------------------------------------------------------------
    first_traj_path = next(iter(trajs.values()))
    mocap_src = first_traj_path / "mav0" / "mocap0" / "data.csv"
    mocap_dst = out_root / "mocap0" / "data.csv"
    if mocap_src.exists():
        mocap_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mocap_src, mocap_dst)
        print(f"[mocap] Copied ground truth -> {mocap_dst}")

    # ------------------------------------------------------------------
    # Per-traj, per-condition
    # ------------------------------------------------------------------
    for traj_name, traj_path in trajs.items():
        print(f"\n[{traj_name}] source: {traj_path}")
        cam0_src = traj_path / "mav0" / "cam0" / "data"
        cam1_src = traj_path / "mav0" / "cam1" / "data"
        traj_out = out_root / "sequence" / traj_name

        # Mono trajectory placeholder
        mono_traj = traj_out / "mono_trajectory.txt"
        if not mono_traj.exists():
            mono_traj.parent.mkdir(parents=True, exist_ok=True)
            mono_traj.touch()

        traj_conditions = []

        for cond in conditions:
            if cond not in DEGRADATIONS:
                print(f"  [WARN] Unknown condition '{cond}', skipping.")
                continue

            print(f"  [{cond}] generating...")
            cond_dir = traj_out / "degradations" / cond
            cond_cam0 = cond_dir / "cam0"
            cond_cam1 = cond_dir / "cam1"
            degrade_fn = DEGRADATIONS[cond]

            # --- cam0: always clean copy ---
            if not cond_cam0.exists() or not any(cond_cam0.glob("*.png")):
                print(f"    cam0: copying clean...")
                _copy_images(cam0_src, cond_cam0)

            # --- cam1: clean copy or degraded ---
            n_existing = len(list(cond_cam1.glob("*.png"))) if cond_cam1.exists() else 0
            n_original = len(list(cam1_src.glob("*.png")))

            if n_existing >= n_original:
                print(f"    cam1: already complete, skipping.")
            elif degrade_fn is None:
                print(f"    cam1: copying clean...")
                _copy_images(cam1_src, cond_cam1)
            else:
                print(f"    cam1: applying {cond}...")
                _degrade_images(cam1_src, cond_cam1, degrade_fn, rng)

            # --- trajectory.txt placeholder ---
            traj = cond_dir / "trajectory.txt"
            if not traj.exists():
                traj.parent.mkdir(parents=True, exist_ok=True)
                traj.touch()

            # --- metadata.yaml ---
            meta = {
                "traj": traj_name,
                "condition": cond,
                "cam0": "clean",
                "cam1": "clean" if degrade_fn is None else cond,
                "n_frames": n_original,
            }
            _write_yaml(cond_dir / "metadata.yaml", meta)
            traj_conditions.append(cond)

        index_entries[traj_name] = traj_conditions

    # ------------------------------------------------------------------
    # dataset_index.yaml
    # ------------------------------------------------------------------
    _write_yaml(out_root / "dataset_index.yaml", {
        "seed": SEED,
        "trajs": index_entries,
    })
    print(f"\nDone. Dataset written to {out_root}")
    return out_root


# ---------------------------------------------------------------------------
# Legacy wrapper (backward compat for old run_pipeline.py sigma-based calls)
# ---------------------------------------------------------------------------

def generate_noisy_dataset(sigma):
    """Legacy: generate EuRoC dataset with Gaussian noise at given sigma."""
    if sigma == 0:
        return ORIG_DATASET
    condition = f"gauss_{sigma}sig"
    # If this sigma isn't in the registry, add it dynamically
    if condition not in DEGRADATIONS:
        DEGRADATIONS[condition] = lambda img, rng, s=sigma: add_gaussian_noise(img, s, rng)
    return generate_noisy_dataset_euroc(condition, ORIG_DATASET, NOISY_BASE)