from pathlib import Path

# ── Project layout ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ORB-SLAM3
ORBSLAM_DIR   = PROJECT_ROOT / "ORB_SLAM3"
VOCAB         = ORBSLAM_DIR / "Vocabulary" / "ORBvoc.txt"

STEREO_BIN      = ORBSLAM_DIR / "Examples" / "Stereo-Inertial" / "stereo_inertial_tum_vi"
STEREO_SETTINGS = ORBSLAM_DIR / "Examples" / "Stereo-Inertial" / "TUM-VI.yaml"

MONO_BIN      = ORBSLAM_DIR / "Examples" / "Monocular-Inertial" / "mono_inertial_tum_vi"
MONO_SETTINGS = ORBSLAM_DIR / "Examples" / "Monocular-Inertial" / "TUM-VI.yaml"

# ── Data paths ──
DATA_DIR     = PROJECT_ROOT / "data"
ORIG_DATASET = DATA_DIR / "TUM_original" / "dataset-room1_512_16"
# TIMESTAMPS   = DATA_DIR / "TUM_original" / "TUM_TimeStamps" / "dataset-room1_512.txt"
TIMESTAMPS = DATA_DIR / "TUM_original" / "dataset-room1_512_16" / "timestamps.txt"
GT_CSV       = ORIG_DATASET / "mav0" / "mocap0" / "data.csv"
IMU_CSV      = ORIG_DATASET / "mav0" / "imu0" / "data.csv"

# Legacy noise pipeline
NOISY_BASE   = DATA_DIR / "noisy_datasets"

# ── Defaults (overridable by experiment YAML) ──
RESULTS_DIR  = PROJECT_ROOT / "results"
SEED         = 42
TIMEOUT      = 1800

# Legacy (used by plot.py / old pipeline)
SIGMAS       = [0, 5, 10, 20, 40, 80]