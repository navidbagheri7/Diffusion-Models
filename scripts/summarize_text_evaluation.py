"""Matched-caption CLIP differences and CFG sensitivity, without ranking claims."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffusion_lab.distribution_metrics import mean_interval
from diffusion_lab.utils import save_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folders = [ROOT / "outputs" / name for name in ["coco_text_base", "coco_text_lora"]]
    baseline, adapted = [json.loads((p / "metrics.json").read_text()) for p in folders]
    assert (folders[0] / "prompts.json").read_bytes() == (folders[1] / "prompts.json").read_bytes()
    for key in [
        "seed",
        "steps",
        "cfg_scale",
        "generated_count",
        "base_identity",
        "assessor_identity",
        "batch_seeds",
    ]:
        assert baseline[key] == adapted[key], key
    before, after = [np.asarray(r["per_example_clip_cosine"]) for r in [baseline, adapted]]
    assert len(before) == len(after) == 256
    differences = after - before
    rows = []
    for name in [
        "coco_text_cfg0",
        "coco_text_cfg1",
        "coco_text_cfg3",
        "coco_text_lora",
        "coco_text_cfg75",
    ]:
        folder = ROOT / "outputs" / name
        report = json.loads((folder / "metrics.json").read_text())
        assert (folder / "prompts.json").read_bytes() == (folders[1] / "prompts.json").read_bytes()
        assert report["seed"] == adapted["seed"] and report["steps"] == adapted["steps"]
        rows.append(
            {
                "run": name,
                "cfg_scale": report["cfg_scale"],
                "clip_cosine": mean_interval(report["per_example_clip_cosine"]),
                "paired_minus_mismatched_mean": report["paired_minus_mismatched_mean"],
                "nfe_per_image": report["nfe_per_image"],
            }
        )
    result = {
        "caption_count": len(before),
        "baseline_clip_cosine": mean_interval(before),
        "adapted_clip_cosine": mean_interval(after),
        "paired_difference_clip_cosine": mean_interval(differences),
        "fraction_of_captions_with_higher_cosine": float((differences > 0).mean()),
        "per_caption_difference": differences.tolist(),
        "cfg_sweep": rows,
        "checkpoint_sha256": adapted["checkpoint_sha256"],
        "scope": "Matched caption/noise diagnostics from one fine-tuning run. Intervals describe variation across captions, not training-seed or model-selection uncertainty. CLIP cosine does not establish counting, spatial binding or perceptual fidelity.",
    }
    folder = ROOT / "outputs/text_alignment_study"
    save_json(folder / "metrics.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(differences, bins=25)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set(
        xlabel="Adapted minus pretrained CLIP cosine",
        ylabel="Caption count",
        title="Identical captions and initial noise",
    )
    axes[1].plot(
        [r["cfg_scale"] for r in rows], [r["clip_cosine"]["mean"] for r in rows], marker="o"
    )
    axes[1].set(
        xlabel="CFG scale", ylabel="Independent CLIP cosine", title="One frozen adapted checkpoint"
    )
    fig.tight_layout()
    fig.savefig(folder / "matched_alignment.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != "per_caption_difference"}, indent=2))


if __name__ == "__main__":
    main()
