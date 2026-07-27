# Contributing

## Codebase overview

This repo contains two parallel pipelines that share a common degradation library. Both pipelines apply identical image degradations to stereo camera datasets, but output in different formats for different downstream tasks.

### Directory structure

```
eece5554_final_project/
├── run_pipeline.py              # ORB-SLAM experiment runner
├── run_ml.py                    # ML dataset generation runner
├── experiments/                 # YAML experiment configs
│   ├── orbslam_blur.yaml
│   └── ml_blur.yaml
├── src/
│   ├── config.py                # Path constants and defaults
│   ├── noise.py                 # Degradation registry + both generators
│   ├── experiment.py            # YAML loaders and config dataclasses
│   ├── slam.py                  # ORB-SLAM3 stereo/mono wrappers
│   ├── evaluate.py              # evo_ape evaluation
│   ├── convert.py               # Trajectory format converters
│   └── plot.py                  # Grouped bar chart + summary CSV
├── data/
│   ├── TUM_original/            # Source EuRoC-format datasets
│   ├── noisy_datasets/          # ORB-SLAM pipeline output
│   └── ml_datasets/             # ML pipeline output
├── results/                     # Per-experiment results
│   └── {experiment_name}/
│       ├── manifest.yaml
│       ├── summary.csv
│       └── {seq}/{cond}/{mode}/
└── ORB_SLAM3/                   # ORB-SLAM3 build
```

### Shared degradation registry

Both pipelines are connected through the `DEGRADATIONS` dict in `src/noise.py`. This is a mapping from string keys to functions with signature `(img, rng) -> img`. The string key (e.g. `"blur_ks7"`) is used in YAML configs for both pipelines and guarantees the same degradation function is applied regardless of which pipeline runs it.

To add a new degradation, add one entry to the dict:

```python
DEGRADATIONS = {
    ...
    "my_new_noise": lambda img, rng: my_function(img, param, rng),
}
```

Both pipelines pick it up immediately. Use it in any experiment YAML by referencing the key in the `conditions` list.

### Pipeline 1: ORB-SLAM (`run_pipeline.py`)

Measures how image degradations affect ORB-SLAM3 localization accuracy.

```
python run_pipeline.py --config experiments/orbslam_blur.yaml
```

**What it does:**

1. Reads the experiment YAML and validates condition keys against the degradation registry.
2. For each (sequence, condition) pair, generates a full EuRoC-format dataset with degraded right camera images (`cam1`). Output goes to `data/noisy_datasets/`.
3. Runs ORB-SLAM3 (stereo-inertial, mono-inertial, or both) on each generated dataset.
4. Evaluates each run with `evo_ape` (absolute trajectory error). Caches results in `stats.yaml` per run for automatic resume.
5. Produces a grouped bar chart and summary CSV in `results/{experiment_name}/`.

**Config schema (`experiments/orbslam_*.yaml`):**

```yaml
name: experiment_tag
seed: 42
timeout: 600
sequences:
  traj1: data/TUM_original/dataset-traj1_512_16
conditions:
  - clean
  - blur_ks7
slam_modes:
  - stereo
  - mono
```

### Pipeline 2: ML dataset generation (`run_ml.py`)

Generates degraded datasets in a flat directory structure suited for ML training.

```
python run_ml.py --config experiments/ml_blur.yaml
```

**What it does:**

1. Reads the experiment YAML and validates condition keys.
2. For each (sequence, condition) pair, copies clean `cam0` images and applies the degradation to `cam1` images. Output goes to `data/ml_datasets/`.
3. Writes `metadata.yaml` per condition and `dataset_index.yaml` at the dataset root.

**Output structure:**

```
data/ml_datasets/{name}/
├── mocap0/
│   └── data.csv
├── sequence/
│   └── {seq_name}/
│       ├── mono_trajectory.txt
│       └── degradations/
│           ├── clean/
│           │   ├── cam0/
│           │   ├── cam1/
│           │   ├── trajectory.txt
│           │   └── metadata.yaml
│           ├── blur_ks7/
│           │   ├── cam0/
│           │   ├── cam1/
│           │   ├── trajectory.txt
│           │   └── metadata.yaml
│           └── ...
└── dataset_index.yaml
```

**Config schema (`experiments/ml_*.yaml`):**

```yaml
name: experiment_tag
seed: 42
sequences:
  traj1: data/TUM_original/dataset-traj1_512_16
conditions:
  - clean
  - blur_ks7
  - gauss_16sig
```

### Resume behavior

Both pipelines support resuming interrupted runs. The ORB-SLAM pipeline checks for existing `stats.yaml` per (sequence, condition, mode) combo and skips completed runs. The ML pipeline checks image counts per condition directory. You can safely kill either pipeline and restart with the same config.

### Adding new degradation types

1. If needed, write a new primitive function in `src/noise.py` following the existing pattern (takes an image and any parameters, returns the degraded image).
2. Add an entry to `DEGRADATIONS` with a descriptive key. Combined degradations chain existing primitives.
3. Reference the new key in any experiment YAML under `conditions`.