"""Evaluate completed checkpoints on fixed seeds and measured NFE budgets.

Waits for training_complete.json so the saved weights are stable. Use --now
only to evaluate an immutable interim checkpoint supplied through a separate run.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--now", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = [
        ("mnist_improved_ddpm", "mnist_ddpm1000", "ddpm", 1000),
        ("mnist_improved_ddpm", "mnist_ddim20", "ddim", 20),
        ("mnist_improved_ddpm", "mnist_ddim50", "ddim", 50),
        ("mnist_improved_ddpm", "mnist_dpm20", "dpm_2m", 20),
        ("mnist_improved_ddpm", "mnist_dpmpp20", "dpmpp_2m", 20),
        ("mnist_edm_quality", "mnist_edm18", "heun", 18),
        ("mnist_edm_quality", "mnist_edm_euler18", "euler", 18),
        ("mnist_consistency_distilled", "mnist_cd1", None, 1),
        ("mnist_consistency_distilled", "mnist_cd2", None, 2),
        ("mnist_consistency_distilled", "mnist_cd4", None, 4),
        ("mnist_consistency_training", "mnist_ct1", None, 1),
        ("mnist_consistency_training", "mnist_ct4", None, 4),
    ]
    while jobs:
        progress = False
        for job in jobs.copy():
            run, name, method, steps = job
            source = root / "outputs" / run
            destination = root / "outputs" / name
            if (destination / "metrics.json").exists():
                jobs.remove(job)
                progress = True
                continue
            if not (source / "checkpoint.pt").exists() or (
                not args.now and not (source / "training_complete.json").exists()
            ):
                continue
            destination.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "diffusion_lab.image_evaluate",
                "--checkpoint",
                str(source / "checkpoint.pt"),
                "--output",
                str(destination),
                "--device",
                args.device,
                "--samples",
                str(args.samples),
                "--scale",
                "1",
                "--steps",
                str(steps),
                "--feature-classifier",
                str(root / "outputs/mnist_classifier/classifier.pt"),
            ]
            if method:
                command.extend(["--method", method])
            with (destination / "process.log").open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({"comparison_completed": name}), flush=True)
            jobs.remove(job)
            progress = True
        if jobs and not progress:
            time.sleep(30)
    subprocess.run([sys.executable, str(root / "scripts/export_assets.py")], cwd=root, check=True)


if __name__ == "__main__":
    main()
