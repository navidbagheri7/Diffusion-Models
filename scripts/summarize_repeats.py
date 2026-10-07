"""Summarize independent generation seeds; split/subset SD is not a seed CI."""

import argparse
import json
from pathlib import Path

from diffusion_lab.distribution_metrics import mean_interval
from diffusion_lab.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((path / "metrics.json").read_text()) for path in args.experiment]
    fixed = ["checkpoint_sha256", "generated_count", "real_count", "steps", "cfg_scale", "sampler"]
    if len(rows) < 2 or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("Need at least two distinct independent generation seeds")
    if any(any(row.get(key) != rows[0].get(key) for key in fixed) for row in rows):
        raise ValueError(
            "Repeated seeds must use identical weights, counts and generation settings"
        )
    keys = ["fid_2048", "kid_mean", "inception_score_mean", "sampling_seconds"]
    save_json(
        args.output,
        {
            "protocol": {key: rows[0].get(key) for key in fixed},
            "seeds": [row["seed"] for row in rows],
            "scope": "Student-t intervals across generation seeds with a fixed held-out reference; excludes dataset-sampling/training uncertainty",
            "metrics": {
                key: mean_interval([row[key] for row in rows]) for key in keys if key in rows[0]
            },
        },
    )


if __name__ == "__main__":
    main()
