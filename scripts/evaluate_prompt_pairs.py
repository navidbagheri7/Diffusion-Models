"""Assess saved matched-noise caption contrasts with independent local CLIP."""

import argparse
import json
from pathlib import Path

import torch

from diffusion_lab.text_evaluate import clip_alignment
from diffusion_lab.utils import save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="JSON list of two prompt indices")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    seed_everything(123, 2)
    metadata = json.loads((args.run / "metrics.json").read_text())
    if metadata["batch_seeds"] != "fixed seed with identical initial noise per prompt":
        raise ValueError("Controlled prompt contrasts require matched initial noise")
    captions = json.loads((args.run / "prompts.json").read_text())["captions"]
    pairs = json.loads(args.pairs.read_text())
    images = torch.load(args.run / "samples.pt", map_location="cpu", weights_only=True, mmap=True)
    report = clip_alignment(images, captions, args.clip, args.device, contrast_pairs=pairs)
    report.update(
        {
            "generation_checkpoint_sha256": metadata["checkpoint_sha256"],
            "generation_weights": metadata["weights"],
            "generation_seed": metadata["seed"],
            "source_run": str(args.run),
            "generated_count": len(images),
        }
    )
    save_json(args.run / "contrast_alignment.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "assessor_identity"}, indent=2))


if __name__ == "__main__":
    main()
