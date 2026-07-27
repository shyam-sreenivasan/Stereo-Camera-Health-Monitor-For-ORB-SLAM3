import os
import subprocess
from pathlib import Path
from src.config import (
    ORBSLAM_DIR, VOCAB, STEREO_SETTINGS, MONO_SETTINGS,
    STEREO_BIN, MONO_BIN, TIMEOUT,
)


def get_env():
    """Build environment with LD_LIBRARY_PATH for ORB-SLAM3."""
    env = os.environ.copy()
    lib_paths = [
        str(ORBSLAM_DIR / "lib"),
        str(ORBSLAM_DIR / "Thirdparty" / "DBoW2" / "lib"),
        str(ORBSLAM_DIR / "Thirdparty" / "g2o" / "lib"),
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths + [existing])
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def _seq_files(orig_dir: Path):
    """Resolve timestamps.txt and imu0/data.csv from the original sequence dir."""
    timestamps = orig_dir / "timestamps.txt"
    imu_csv    = orig_dir / "mav0" / "imu0" / "data.csv"
    return timestamps, imu_csv


def run_stereo(dataset_dir, tag, run_dir, orig_dir=None):
    """Run stereo-inertial TUM-VI example."""
    cam0 = dataset_dir / "mav0" / "cam0" / "data"
    cam1 = dataset_dir / "mav0" / "cam1" / "data"
    timestamps, imu_csv = _seq_files(orig_dir or dataset_dir)

    cmd = [
        str(STEREO_BIN),
        str(VOCAB), str(STEREO_SETTINGS),
        str(cam0), str(cam1),
        str(timestamps), str(imu_csv),
        tag,
    ]

    print(f"    Running stereo-inertial...")
    with open(run_dir / "log.txt", "w") as log:
        try:
            subprocess.run(
                cmd, cwd=str(run_dir), env=get_env(),
                stdout=log, stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {TIMEOUT}s")


def run_mono(dataset_dir, tag, run_dir, orig_dir=None):
    """Run mono-inertial TUM-VI example."""
    cam0 = dataset_dir / "mav0" / "cam0" / "data"
    timestamps, imu_csv = _seq_files(orig_dir or dataset_dir)

    cmd = [
        str(MONO_BIN),
        str(VOCAB), str(MONO_SETTINGS),
        str(cam0),
        str(timestamps), str(imu_csv),
        tag,
    ]

    print(f"    Running mono-inertial...")
    with open(run_dir / "log.txt", "w") as log:
        try:
            subprocess.run(
                cmd, cwd=str(run_dir), env=get_env(),
                stdout=log, stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {TIMEOUT}s")
