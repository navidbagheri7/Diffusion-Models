"""Compare CFG/noisy classifier guidance, including after final RGB training.

Every row requests the same class and uses seed 123. An unconditional row is
evaluated against that requested class deliberately: its class agreement is a
baseline, not its unconditional distribution quality.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--class-label", type=int, default=3)
    parser.add_argument("--wait-final", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assessment-classifier", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs" / f"{args.data}_edm_quality"
    checkpoint = args.checkpoint or source / "checkpoint.pt"
    classifier = root / "outputs" / f"{args.data}_classifier/classifier.pt"
    assessment = args.assessment_classifier or classifier
    while (
        not checkpoint.exists()
        or not classifier.exists()
        or not assessment.exists()
        or (args.wait_final and not (source / "training_complete.json").exists())
    ):
        time.sleep(30)
    cases = [("unconditional", 0, None), ("conditional", 1, None), ("cfg", 1.5, None)]
    cases += [
        ("cg_low", 0, 5 if args.data == "mnist" else 3),
        ("cg_high", 0, 20 if args.data == "mnist" else 10),
    ]
    for name, cfg_scale, cg_scale in cases:
        output = root / "outputs" / f"{args.data}_guidance_{name}"
        output.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "diffusion_lab.image_evaluate",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            args.device,
            "--samples",
            str(args.samples),
            "--class-label",
            str(args.class_label),
            "--steps",
            "50",
            "--scale",
            str(cfg_scale),
            "--feature-classifier",
            str(assessment),
        ]
        if cg_scale is not None:
            command += ["--classifier", str(classifier), "--classifier-scale", str(cg_scale)]
        with (output / "process.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"guidance_example_completed": name}), flush=True)
    for script in ["build_guidance_gallery.py", "export_assets.py", "build_advanced_gallery.py"]:
        subprocess.run([sys.executable, str(root / "scripts" / script)], cwd=root, check=True)


if __name__ == "__main__":
    main()
