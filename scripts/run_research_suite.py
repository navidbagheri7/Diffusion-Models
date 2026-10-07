"""Run independent, bounded teaching experiments across explicitly chosen GPUs.

Run from the repository root. Each process has one visible GPU and its own log.
Existing completed runs are skipped unless --rerun is specified.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GMM = [
    ("gmm_vp_epsilon", ["experiment=gmm_vp", "steps=6000"]),
    ("gmm_ve_score", ["experiment=gmm_ve", "steps=6000"]),
    ("gmm_linear_epsilon", ["process.schedule=linear", "process.prediction=epsilon", "steps=6000"]),
    ("gmm_cosine_v", ["process.prediction=v", "steps=6000"]),
    ("gmm_flow_ot", ["experiment=gmm_flow", "process.coupling=ot", "batch_size=128", "steps=6000"]),
    (
        "gmm_flow_trigonometric",
        ["experiment=gmm_flow", "process.flow_path=trigonometric", "steps=6000"],
    ),
    ("gmm_reflow", ["experiment=gmm_reflow", "steps=6000"]),
    ("gmm_bridge_sde", ["experiment=gmm_bridge", "steps=8000"]),
    (
        "gmm_bridge_ode",
        ["experiment=gmm_bridge", "process.prediction=velocity", "sampler=ode", "steps=8000"],
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = [
        job
        for job in GMM
        if args.rerun or not (root / "outputs" / job[0] / "metrics.json").exists()
    ]

    def lane(index):
        gpu = args.gpus[index]
        for name, overrides in jobs[index :: len(args.gpus)]:
            directory = root / "outputs" / name
            directory.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "diffusion_lab.train",
                *overrides,
                "device=cuda:0",
                f"run_dir=outputs/{name}",
            ]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            print(f"START GPU {gpu}: {name}", flush=True)
            with (directory / "process.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            metrics = json.loads((directory / "metrics.json").read_text())
            print(f"DONE GPU {gpu}: {name}; sliced W2={metrics['sliced_w2']:.4f}", flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        list(executor.map(lane, range(len(args.gpus))))


if __name__ == "__main__":
    main()
