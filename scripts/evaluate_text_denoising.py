"""Matched held-out caption, empty-caption and shuffled-caption denoising tests.

Shared latent posterior draws, times and noise isolate the conditioning choice.
This is a denoising diagnostic, not a generated-image quality metric.
"""

import argparse
import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf

from diffusion_lab.caption_data import CaptionDataset
from diffusion_lab.distribution_metrics import mean_interval
from diffusion_lab.text import load_trainable_state
from diffusion_lab.text_train import (
    encode_latents,
    encode_text,
    load_components,
    model_fingerprint,
    native_prediction,
)
from diffusion_lab.utils import save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=125)
    parser.add_argument("--split", choices=["validation", "evaluation"], default="evaluation")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size < 2:
        raise ValueError("At least two captions per batch are needed to shuffle conditioning")
    seed_everything(args.seed, 4)
    device = torch.device(args.device)
    with args.checkpoint.open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
        handle.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if checkpoint.get("format") != "text_v1":
        raise ValueError("Expected a text_train checkpoint")
    cfg = OmegaConf.create(checkpoint["config"])
    if model_fingerprint(cfg.base_model) != checkpoint["base_identity"]:
        raise ValueError("Codec/text/base content differs from the training checkpoint")
    model, vae, encoder, tokenizer, scheduler = load_components(cfg, device)
    load_trainable_state(model, checkpoint["model" if args.latest else "ema"])
    model.eval().requires_grad_(False)
    data = CaptionDataset(cfg.data_root, args.split, cfg.resolution, args.samples)
    # Globally roll captions, including batches with only one final example.
    captions = [row["captions"][0] for row in data.rows]
    shuffled = captions[-1:] + captions[:-1]
    levels = [100, 300, 500, 700, 900, 999]
    values = {time: {name: [] for name in ["caption", "empty", "shuffled"]} for time in levels}
    with torch.no_grad():
        for offset in range(0, len(data), args.batch_size):
            images = torch.stack(
                [data[i][0] for i in range(offset, min(offset + args.batch_size, len(data)))]
            ).to(device)
            latents = encode_latents(vae, images)
            noise = torch.randn_like(latents)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=cfg.amp and device.type == "cuda",
            ):
                contexts = {
                    "caption": encode_text(
                        encoder, tokenizer, captions[offset : offset + len(images)], device
                    ),
                    "empty": encode_text(encoder, tokenizer, [""] * len(images), device),
                    "shuffled": encode_text(
                        encoder, tokenizer, shuffled[offset : offset + len(images)], device
                    ),
                }
                for level in levels:
                    times = torch.full((len(images),), level, dtype=torch.long, device=device)
                    noisy = scheduler.add_noise(latents, noise, times)
                    target = (
                        noise
                        if scheduler.config.prediction_type == "epsilon"
                        else scheduler.get_velocity(latents, noise, times)
                    )
                    for name, context in contexts.items():
                        prediction = native_prediction(model, cfg, noisy, times, context)
                        mse = (prediction.float() - target).square().flatten(1).mean(1)
                        values[level][name].extend(mse.cpu().tolist())
    rows = []
    for time, conditions in values.items():
        caption = torch.tensor(conditions["caption"])
        rows.append(
            {
                "discrete_timestep": time,
                "mse": {name: mean_interval(rows) for name, rows in conditions.items()},
                "paired_empty_minus_caption": mean_interval(
                    (torch.tensor(conditions["empty"]) - caption).tolist()
                ),
                "paired_shuffled_minus_caption": mean_interval(
                    (torch.tensor(conditions["shuffled"]) - caption).tolist()
                ),
            }
        )
    save_json(
        args.output / "metrics.json",
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": digest.hexdigest(),
            "training_steps": checkpoint["step"],
            "weights": "latest" if args.latest else "ema",
            "samples": len(data),
            "split": args.split,
            "seed": args.seed,
            "image_ids": [row["image_id"] for row in data.rows],
            "manifest_sha256": data.manifest_sha256,
            "prediction_type": scheduler.config.prediction_type,
            "results": rows,
            "interpretation": "Positive paired differences indicate better denoising with the paired caption. They do not establish generated-image fidelity or counting accuracy.",
        },
    )
    save_json(args.output / "per_example.json", values)


if __name__ == "__main__":
    main()
