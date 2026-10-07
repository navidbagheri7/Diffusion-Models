"""Select refinement schedules separately from the final matched comparison.

Both checkpoints use the same reference subset, labels and initial-noise seed.
The selection subset chooses sigma levels/clipping separately for each call
budget; a disjoint reference subset and new seed supply the final metrics.
This is a development comparison, not a pristine CIFAR benchmark.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_lab.utils import save_json

ROOT = Path(__file__).resolve().parents[1]


def evaluate(checkpoint, output, sigmas, clipped, args, final=False):
    output.mkdir(parents=True, exist_ok=True)
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
        "--batch-size",
        str(args.batch_size),
        "--samples",
        str(args.final_samples if final else args.selection_samples),
        "--reference-offset",
        str(args.selection_samples if final else 0),
        "--scale",
        "1",
        "--seed",
        str(args.final_seed if final else args.selection_seed),
        "--consistency-sigmas",
        *map(str, sigmas),
    ]
    if clipped:
        command.append("--clip-consistency")
    save_json(output / "command.json", command)
    with (output / "execution.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    row = json.loads((output / "metrics.json").read_text())
    print(
        json.dumps(
            {"case": output.name, "steps": row["steps"], "fid": row["fid_2048"], "clipped": clipped}
        ),
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--selection-samples", type=int, default=1000)
    parser.add_argument("--final-samples", type=int, default=5000)
    parser.add_argument("--selection-seed", type=int, default=123)
    parser.add_argument("--final-seed", type=int, default=124)
    args = parser.parse_args()
    if min(args.batch_size, args.selection_samples, args.final_samples) <= 0:
        raise ValueError("Positive sample counts and batch size required")
    if args.selection_seed == args.final_seed:
        raise ValueError("Selection and final generation seeds must differ")
    args.output.mkdir(parents=True, exist_ok=True)
    schedules = [[80], *[[80, level] for level in [0.3, 1, 3, 10]], [80, 3, 1, 0.3], [80, 10, 3, 1]]
    results, selected = {}, {}
    # Freeze both inputs before evaluating; training can safely continue elsewhere.
    for family, original in [("baseline", args.baseline), ("candidate", args.candidate)]:
        frozen = args.output / f"{family}.pt"
        digest = hashlib.sha256()
        with original.open("rb") as source, frozen.open("wb") as destination:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                destination.write(chunk)
        rows = []
        for index, sigmas in enumerate(schedules):
            for clipped in [False] if len(sigmas) == 1 else [False, True]:
                folder = args.output / f"{family}_select_{index:02d}_{int(clipped)}"
                row = evaluate(frozen, folder, sigmas, clipped, args)
                assert row["checkpoint_sha256"] == digest.hexdigest()
                rows.append(row)
        selected[family] = rows
        winners = [
            min((row for row in rows if row["steps"] == calls), key=lambda row: row["fid_2048"])
            for calls in [1, 2, 4]
        ]
        finals = []
        for winner in winners:
            folder = args.output / f"{family}_final_{winner['steps']}"
            finals.append(
                evaluate(
                    frozen,
                    folder,
                    winner["consistency_sigmas"],
                    winner["clip_consistency"],
                    args,
                    final=True,
                )
            )
        results[family] = finals
        save_json(args.output / "progress.json", {"selection": selected, "final": results})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for family, rows in results.items():
        for axis, key in zip(axes, ["fid_2048", "inception_score_mean"], strict=True):
            axis.plot(
                [row["nfe_per_image"] for row in rows],
                [row[key] for row in rows],
                marker="o",
                label=family,
            )
            axis.set(xlabel="Actual denoiser calls", ylabel="FID" if key == "fid_2048" else "IS")
            axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "comparison.png", dpi=180)
    plt.close(figure)
    save_json(
        args.output / "metrics.json",
        {
            "study": "Matched baseline versus retrained consistency refinement",
            "selection": selected,
            "final": results,
            "limitations": [
                "Single training seed; configurations differ in loss weighting and target/inference EMA.",
                "Their individual causal contributions are not isolated by this comparison.",
                "CIFAR test data have already been used during repository development.",
                "FID is biased at finite sample counts; compare within this matched protocol.",
                "Clipping before re-noising is a sampler choice, not a free quality guarantee.",
            ],
        },
    )
    save_json(args.output / "completed.json", {"status": "All selection and final cases completed"})


if __name__ == "__main__":
    main()
