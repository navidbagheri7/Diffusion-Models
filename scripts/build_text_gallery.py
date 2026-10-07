"""Label controlled prompt contrasts and CFG examples from actual saved outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion_lab.metrics import image_pixels
from diffusion_lab.utils import save_json

ROOT = Path(__file__).resolve().parents[1]


def first_images(name, count):
    path = ROOT / "outputs" / name
    samples = torch.load(path / "samples.pt", weights_only=True, map_location="cpu", mmap=True)
    report = json.loads((path / "metrics.json").read_text())
    prompts = json.loads((path / "prompts.json").read_text())
    return image_pixels(samples[:count]).permute(0, 2, 3, 1).numpy(), report, prompts


def main():
    target = ROOT / "assets/studies/text_gallery"
    target.mkdir(parents=True, exist_ok=True)
    base, br, bp = first_images("coco_text_base_pairs", 6)
    adapted, ar, ap = first_images("coco_text_lora_pairs", 6)
    assert bp == ap and br["seed"] == ar["seed"] and br["cfg_scale"] == ar["cfg_scale"]
    labels = ["Red bus", "Blue bus", "Two dogs", "One dog", "Apples", "Oranges"]
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for row, images in enumerate([base, adapted]):
        for column, pixels in enumerate(images):
            ax = axes[row, column]
            ax.imshow(pixels)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(labels[column], fontsize=12)
            if column == 0:
                ax.set_ylabel("Pretrained base" if row == 0 else "COCO LoRA", fontsize=12)
    fig.suptitle(
        "Controlled text contrasts: identical initial noise, 30 DDIM steps, CFG 5", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(target / "prompt_contrasts.png", dpi=160)
    plt.close(fig)
    cases = [
        "coco_text_cfg0",
        "coco_text_cfg1",
        "coco_text_cfg3",
        "coco_text_lora",
        "coco_text_cfg75",
    ]
    fig, axes = plt.subplots(1, len(cases), figsize=(15, 4))
    reports, prompt = [], None
    for ax, name in zip(axes, cases, strict=True):
        images, report, captions = first_images(name, 1)
        if prompt is None:
            prompt = captions["captions"][0]
        assert captions["captions"][0] == prompt
        ax.imshow(images[0])
        ax.axis("off")
        ax.set_title(f"CFG {report['cfg_scale']:g}\n{report['nfe_per_image']} calls", fontsize=12)
        reports.append(report)
    fig.suptitle(prompt + "\nFixed noise; CFG 0 uses the empty-prompt branch", fontsize=12)
    fig.tight_layout()
    fig.savefig(target / "text_cfg.png", dpi=160)
    plt.close(fig)
    save_json(
        target / "provenance.json",
        {
            "prompt_contrasts": {"baseline": br, "adapted": ar, "captions": bp["captions"][:6]},
            "cfg_cases": [
                {"run": name, "report": report} for name, report in zip(cases, reports, strict=True)
            ],
            "selection": "The first six predeclared controlled prompts, and the first held-out caption. No image selection for visual quality.",
        },
    )
    print("Built labeled prompt and guidance figures from saved model outputs.")


if __name__ == "__main__":
    main()
