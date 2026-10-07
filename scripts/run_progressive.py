"""Train and evaluate a reproducible DDIM 32 -> 16 -> 8 -> 4 -> 2 -> 1 chain.

Every stage keeps its own optimizer/checkpoints/config and freezes the completed
teacher. This script can resume an interrupted chain without shortening budgets.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command, directory, root):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "process.log").open("a") as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--stage-steps", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("outputs/mnist_progressive"))
    parser.add_argument(
        "--feature-classifier",
        type=Path,
        default=Path("outputs/mnist_assessment_classifier/classifier.pt"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = args.teacher.resolve()
    chain = []
    for steps in [16, 8, 4, 2, 1]:
        stage = args.output / f"stage_{steps}"
        if not (stage / "training_complete.json").exists():
            command = [
                sys.executable,
                "-m",
                "diffusion_lab.image_train",
                "--config-name",
                "image_progressive",
                f"teacher={teacher}",
                f"device={args.device}",
                f"steps={args.stage_steps}",
                f"student_steps={steps}",
                "model.width=32",
                "model.channel_mult=[1,2,2]",
                "model.attention_levels=[2]",
                "accumulation=1",
                f"run_dir={stage}",
            ]
            if (stage / "checkpoint.pt").exists():
                command.append(f"resume={stage / 'checkpoint.pt'}")
            run(command, stage, root)
        teacher = (stage / "checkpoint.pt").resolve()
        output = args.output.parent / f"mnist_progressive_{steps}_eval"
        if not (output / "metrics.json").exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "diffusion_lab.image_evaluate",
                    "--checkpoint",
                    str(teacher),
                    "--device",
                    args.device,
                    "--output",
                    str(output),
                    "--steps",
                    str(steps),
                    "--scale",
                    "1",
                    "--samples",
                    str(args.samples),
                    "--feature-classifier",
                    str(args.feature_classifier),
                    "--interpolation",
                ],
                output,
                root,
            )
        chain.append(json.loads((output / "metrics.json").read_text()))
        (args.output / "chain.json").write_text(json.dumps(chain, indent=2) + "\n")
        subprocess.run([sys.executable, "scripts/export_assets.py"], cwd=root, check=True)
    (args.output / "training_complete.json").write_text(
        json.dumps({"stages": 5, "updates_per_stage": args.stage_steps})
    )


if __name__ == "__main__":
    main()
