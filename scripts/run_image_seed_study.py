"""Matched MNIST EDM/flow training seeds with independent digit assessment."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from diffusion_lab.distribution_metrics import mean_interval
from diffusion_lab.utils import save_json
from scripts.run_evidence_studies import ROOT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["edm", "flow"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seeds", nargs="+", type=int, choices=[42, 43, 44], default=[42, 43, 44])
    parser.add_argument("--output", type=Path, default=Path("outputs/mnist_training_seed_study"))
    args = parser.parse_args()
    args.output = args.output.resolve()
    folder = args.output / "configurations"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        name = f"{args.algorithm}_seed{seed}"
        directory = args.output / name
        cfg = OmegaConf.load(ROOT / "outputs/mnist_edm_quality/config.yaml")
        cfg.algorithm = args.algorithm
        cfg.seed = seed
        cfg.device = args.device
        cfg.steps = args.steps
        cfg.run_dir = str(directory)
        cfg.sampling_steps = 16
        cfg.preview_scale = 1.0
        cfg.cpu_threads = 2
        cfg.save_every = 2000
        cfg.preview_every = 5000
        if (directory / "checkpoint.pt").exists() and not (
            directory / "training_complete.json"
        ).exists():
            cfg.resume = str(directory / "checkpoint.pt")
        OmegaConf.save(cfg, folder / f"{name}.yaml", resolve=True)
        if not (directory / "training_complete.json").exists():
            run(
                [
                    "-m",
                    "diffusion_lab.image_train",
                    "--config-path",
                    str(folder),
                    "--config-name",
                    name,
                ],
                directory,
            )
        evaluation = directory / "evaluation"
        if not (evaluation / "metrics.json").exists():
            run(
                [
                    "-m",
                    "diffusion_lab.image_evaluate",
                    "--checkpoint",
                    str(directory / "checkpoint.pt"),
                    "--output",
                    str(evaluation),
                    "--device",
                    args.device,
                    "--samples",
                    "2048",
                    "--batch-size",
                    "64",
                    "--steps",
                    "16",
                    "--scale",
                    "1",
                    "--seed",
                    "123",
                    "--feature-classifier",
                    str(ROOT / "outputs/mnist_assessment_classifier/classifier.pt"),
                ],
                evaluation,
            )
        row = json.loads((evaluation / "metrics.json").read_text())
        row["training_seed"] = seed
        rows.append(row)
        print(f"{name}: completed training and independent evaluation", flush=True)
    # A worker may handle one seed while other workers train the remaining
    # models. Never publish a three-seed summary before all three are evaluated.
    completed = [
        args.output / f"{args.algorithm}_seed{s}/evaluation/metrics.json" for s in [42, 43, 44]
    ]
    if not all(path.exists() for path in completed):
        return
    rows = []
    for seed, path in zip([42, 43, 44], completed, strict=True):
        row = json.loads(path.read_text())
        row["training_seed"] = seed
        rows.append(row)
    numeric = [
        key
        for key in ["frechet_digit_features_128", "requested_class_accuracy"]
        if all(isinstance(r.get(key), (int, float)) for r in rows)
    ]
    save_json(
        args.output / f"{args.algorithm}_summary.json",
        {
            "algorithm": args.algorithm,
            "training_seeds": [42, 43, 44],
            "steps_per_model": args.steps,
            "results": rows,
            "summary": {key: mean_interval([row[key] for row in rows]) for key in numeric},
            "scope": "Independent training seeds; same image U-Net/data/update budget. Fixed evaluation seed 123, 2048 digit references, ordinary class conditioning. EDM 31 versus flow 32 model calls at 16 Heun steps. Digit feature distance is not Inception FID; historical MNIST test use disclosed.",
        },
    )


if __name__ == "__main__":
    main()
