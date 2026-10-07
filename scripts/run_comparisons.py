"""Reproducible same-seed sampler and guidance comparisons on trained GMM runs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = [
        ("gmm_ddpm_ancestral", "gmm_ddpm_x0", ["--method", "ddpm", "--steps", "1000"]),
        ("gmm_ddim_20", "gmm_ddpm_x0", ["--method", "ddim", "--steps", "20"]),
        ("gmm_ddim_5", "gmm_ddpm_x0", ["--method", "ddim", "--steps", "5"]),
        ("gmm_flow_euler_10", "gmm_flow", ["--solver", "euler", "--steps", "10"]),
        ("gmm_flow_heun_10", "gmm_flow", ["--solver", "heun", "--steps", "10"]),
        ("gmm_flow_rk4_10", "gmm_flow", ["--solver", "rk4", "--steps", "10"]),
        ("gmm_reflow_euler_10", "gmm_reflow", ["--solver", "euler", "--steps", "10"]),
    ]
    jobs += [
        (f"cfg_scale_{scale}", "gmm_guidance", ["--class-label", "0", "--scale", str(scale)])
        for scale in [0, 1, 2, 4]
    ]
    jobs += [
        (
            f"classifier_scale_{scale}",
            "gmm_guidance",
            ["--class-label", "0", "--guidance", "classifier", "--scale", str(scale)],
        )
        for scale in [1, 2]
    ]
    jobs += [
        (
            "classifier_analytic_scale_1",
            "gmm_guidance",
            [
                "--class-label",
                "0",
                "--guidance",
                "classifier",
                "--classifier-kind",
                "analytic",
                "--scale",
                "1",
            ],
        )
    ]
    rows = []
    for name, run, options in jobs:
        directory = root / "outputs" / name
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "diffusion_lab.evaluate",
            "--checkpoint",
            f"outputs/{run}/checkpoint.pt",
            "--output",
            f"outputs/{name}",
            "--device",
            args.device,
            "--samples",
            "1024",
            "--seed",
            "123",
            *options,
        ]
        with (directory / "process.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((directory / "metrics.json").read_text())
        rows.append({"experiment": name, **metrics})
        print(f"{name}: sliced W2={metrics['sliced_w2']:.4f}", flush=True)
    output = root / "outputs/comparisons.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
