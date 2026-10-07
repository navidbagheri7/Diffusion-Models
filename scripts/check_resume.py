"""Verify CPU continuation against an uninterrupted run, including augmentation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/resume_verification"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--full-steps", type=int, default=4)
    parser.add_argument("--split-steps", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.split_steps < args.full_steps or args.workers < 0:
        parser.error("Require positive split_steps < full_steps and nonnegative workers")
    common = [
        sys.executable,
        "-m",
        "diffusion_lab.image_train",
        "data=cifar10",
        "model.width=16",
        "model.channel_mult=[1,2]",
        "model.blocks_per_level=1",
        "batch_size=2",
        "accumulation=1",
        "save_every=2",
        "preview_every=100",
        "sampling_steps=2",
        f"workers={args.workers}",
        f"data.train_limit={args.train_limit}",
        "data.validation_limit=16",
        "device=cpu",
        "amp=false",
        "log_every=2",
    ]
    directory = args.output
    if any((directory / name / "checkpoint.pt").exists() for name in ["full", "split"]):
        parser.error("Choose an empty --output directory to preserve earlier verification weights")
    directory.mkdir(parents=True, exist_ok=True)
    for name, steps, extra in [
        ("full", args.full_steps, []),
        ("split", args.split_steps, []),
        ("split", args.full_steps, [f"resume={directory}/split/checkpoint.pt"]),
    ]:
        with (directory / f"{name}_{steps}.log").open("w") as log:
            subprocess.run(
                [*common, f"steps={steps}", f"run_dir={directory}/{name}", *extra],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    states = [
        torch.load(directory / name / "checkpoint.pt", map_location="cpu", weights_only=False)
        for name in ["full", "split"]
    ]
    for field in ["model", "ema"]:
        for key in states[0][field]:
            torch.testing.assert_close(states[0][field][key], states[1][field][key], rtol=0, atol=0)
    (directory / "result.json").write_text(
        json.dumps(
            {
                "cpu_weights_and_ema_bitwise_equal": True,
                "uninterrupted_steps": args.full_steps,
                "split_steps": [args.split_steps, args.full_steps - args.split_steps],
                "workers": args.workers,
                "train_limit": args.train_limit,
                "augmentation": "device horizontal flips with restored RNG",
                "scope": "EDM constant-LR continuation; CUDA nondeterminism and worker timing are separate",
            },
            indent=2,
        )
        + "\n"
    )
    print("Resume verification passed")


if __name__ == "__main__":
    main()
