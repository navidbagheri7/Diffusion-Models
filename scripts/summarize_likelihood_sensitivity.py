"""Publish per-example likelihood diagnostics and matched numerical comparisons."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from diffusion_lab.distribution_metrics import mean_interval
from diffusion_lab.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/expanded_likelihood_study"))
    args = parser.parse_args()
    collected = {}
    for case in sorted(args.root.iterdir()):
        if not case.is_dir() or not (case / "summary.json").exists():
            continue
        values, batches = [], []
        for batch in sorted(case.glob("batch_*")):
            report = json.loads((batch / "metrics.json").read_text())
            raw = json.loads((batch / "per_example.json").read_text())
            assert report["reference_offset"] == len(values)
            if "negative_elbo_bpd" in raw:
                bpd = raw["negative_elbo_bpd"]
            else:
                cfg = OmegaConf.load(batch / "config.yaml")
                dimensions = cfg.data.channels * cfg.data.image_size**2
                bpd = [7 - value / (dimensions * math.log(2)) for value in raw["log_density_nats"]]
            assert len(bpd) == report["samples"]
            values.extend(bpd)
            batches.append(report)
        assert len({row["checkpoint_sha256"] for row in batches}) == 1
        summary = json.loads((case / "summary.json").read_text())
        assert len(values) == summary["samples"]
        assert math.isclose(np.mean(values), summary["bpd"]["mean"], abs_tol=1e-12)
        collected[case.name] = np.asarray(values)
        save_json(
            case / "per_example_bpd.json",
            {
                "case": case.name,
                "bpd": values,
                "batches": batches,
                "ordering": "Contiguous offsets in the deterministic balanced reference order",
                "scope": "Known-label conditional diagnostics. Reused within-batch noise/probe seed; intervals across examples do not measure independent Monte Carlo uncertainty.",
            },
        )
    baseline = collected["cifar_edm"][:128]
    paired = []
    for name in ["cifar_edm_grid200", "cifar_edm_probes4", "cifar_edm_noise124"]:
        values = collected[name]
        assert len(values) == len(baseline)
        differences = values - baseline
        paired.append(
            {
                "case": name,
                "examples": len(values),
                "baseline_bpd": mean_interval(baseline),
                "changed_bpd": mean_interval(values),
                "paired_difference_bpd": mean_interval(differences),
                "mean_absolute_difference_bpd": float(np.abs(differences).mean()),
            },
        )
    save_json(
        args.root / "paired_sensitivity.json",
        {
            "baseline": "First 128 examples, CIFAR EDM, 100 Heun steps, one probe, seed 123",
            "results": paired,
            "scope": "Matched examples and dequantization seeds for grid/probe checks. Changing seed changes both dequantization and trace probes. These comparisons are sensitivity checks, not convergence proofs.",
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ["mnist_elbo", "mnist_edm", "cifar_edm"]:
        axes[0].hist(collected[name], bins=30, density=True, histtype="step", label=name)
    axes[0].set(xlabel="Conditional bits per dimension", ylabel="Example density")
    axes[0].legend()
    labels = [row["case"].removeprefix("cifar_edm_") for row in paired]
    means = [row["paired_difference_bpd"]["mean"] for row in paired]
    errors = np.array(
        [
            [
                m - r["paired_difference_bpd"]["confidence_interval"][0],
                r["paired_difference_bpd"]["confidence_interval"][1] - m,
            ]
            for r, m in zip(paired, means, strict=True)
        ]
    ).T
    axes[1].errorbar(labels, means, yerr=errors, fmt="o", capsize=5)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(
        ylabel="Changed minus matched baseline bpd", title="95% intervals across 128 examples"
    )
    fig.tight_layout()
    fig.savefig(args.root / "likelihood_sensitivity.png", dpi=160)
    plt.close(fig)
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
