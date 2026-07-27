#!/usr/bin/env python3
"""
ML dataset generation pipeline.

Usage:
    python run_ml.py --config experiments/ml_blur.yaml
"""

import argparse

from src.experiment import load_ml_config
from src.noise import generate_ml_dataset


def main():
    parser = argparse.ArgumentParser(description="ML dataset generator")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    cfg = load_ml_config(args.config)

    print(f"Experiment : {cfg.name}")
    print(f"Trajs      : {list(cfg.trajs.keys())}")
    print(f"Conditions : {cfg.conditions}")
    print(f"Output     : {cfg.dataset_out}")
    print()

    generate_ml_dataset(
        trajs=cfg.trajs,
        conditions=cfg.conditions,
        out_root=cfg.dataset_out,
    )


if __name__ == "__main__":
    main()