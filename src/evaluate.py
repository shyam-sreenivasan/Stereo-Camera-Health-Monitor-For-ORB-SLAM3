import json
import os
import subprocess
import zipfile
from src.convert import convert_orbslam_to_tum


def evaluate(run_dir, gt_tum, tag):
    """Run evo_ape and return metrics dict or None on failure."""
    traj_file = run_dir / f"f_{tag}.txt"
    if not traj_file.exists():
        print(f"    No trajectory file found at {traj_file}")
        return None

    traj_tum = run_dir / "traj_tum.txt"
    convert_orbslam_to_tum(traj_file, traj_tum)

    results_zip = run_dir / "ape_result.zip"
    if results_zip.exists():
        results_zip.unlink()

    cmd = [
        "evo_ape", "tum", str(gt_tum), str(traj_tum),
        "--align", "--correct_scale",
        "--save_results", str(results_zip),
    ]

    try:
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, stdin=subprocess.DEVNULL,
        )
        with open(run_dir / "evo_output.txt", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        if result.returncode != 0:
            print(f"    evo_ape failed: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"    evo_ape timed out")
        return None

    if not results_zip.exists():
        print(f"    evo_ape did not produce results")
        return None

    try:
        with zipfile.ZipFile(results_zip) as z:
            with z.open("stats.json") as f:
                stats = json.load(f)
        return stats
    except Exception as e:
        print(f"    Failed to parse evo results: {e}")
        return None