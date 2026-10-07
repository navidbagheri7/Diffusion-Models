"""Text CFG generation, matched base/adapted comparisons, and prompt diagnostics.

The text evaluator is independent of class-label guidance. CLIP cosine is a
coarse alignment diagnostic, not proof of counting/spatial-language correctness.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .caption_data import CaptionDataset
from .metrics import image_metrics, image_pixels
from .text import load_trainable_state, text_guided_prediction
from .text_train import (
    codec_geometry,
    decode_latents,
    encode_text,
    load_components,
    model_fingerprint,
    native_prediction,
)
from .utils import save_json, seed_everything
from .visualize import save_rendered_image


@torch.no_grad()
def generate_text(
    model,
    vae,
    text_encoder,
    tokenizer,
    training_scheduler,
    cfg,
    captions,
    steps=30,
    scale=5.0,
    seed=123,
    matched_noise=False,
):
    from diffusers import DDIMScheduler

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = DDIMScheduler.from_config(training_scheduler.config)
    scheduler.set_timesteps(steps, device=device)
    # Use a separate seed per evaluation batch, recorded by the CLI. Conditions
    # within a batch are compared with identical initial noise in gallery mode.
    channels = vae.config.latent_channels
    factor, _, _ = codec_geometry(vae, cfg.resolution)
    noise = torch.randn(
        1 if matched_noise else len(captions),
        channels,
        cfg.resolution // factor,
        cfg.resolution // factor,
        generator=generator,
        device=device,
    )
    if matched_noise:
        noise = noise.expand(len(captions), -1, -1, -1).clone()
    x = noise * scheduler.init_noise_sigma
    amp = cfg.amp and device.type == "cuda"
    calls = 0
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        context = encode_text(text_encoder, tokenizer, captions, device)
        empty = encode_text(text_encoder, tokenizer, [""] * len(captions), device)
        for time_point in scheduler.timesteps:
            times = time_point.expand(len(x))
            native = scheduler.scale_model_input(x, time_point)
            prediction, count = text_guided_prediction(
                lambda states, times, conditioning: native_prediction(
                    model, cfg, states, times, conditioning
                ),
                native,
                times,
                context,
                empty,
                scale,
            )
            calls += count
            x = scheduler.step(prediction.float(), time_point, x, eta=0).prev_sample
    images = decode_latents(vae, x)
    return images.float().cpu(), calls


@torch.no_grad()
def clip_alignment(images, captions, snapshot, device, batch_size=16, contrast_pairs=None):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(snapshot, local_files_only=True)
    assessor = CLIPModel.from_pretrained(snapshot, local_files_only=True).to(device).eval()
    image_vectors, text_vectors = [], []
    pixels = image_pixels(images)
    for start in range(0, len(images), batch_size):
        batch = processor(
            images=list(pixels[start : start + batch_size]),
            text=captions[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        image_feature = assessor.get_image_features(pixel_values=batch.pixel_values)
        text_feature = assessor.get_text_features(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        )
        image_vectors.append(torch.nn.functional.normalize(image_feature.float(), dim=-1).cpu())
        text_vectors.append(torch.nn.functional.normalize(text_feature.float(), dim=-1).cpu())
    image_vectors, text_vectors = torch.cat(image_vectors), torch.cat(text_vectors)
    paired = (image_vectors * text_vectors).sum(-1)
    mismatched = (image_vectors * text_vectors.roll(1, 0)).sum(-1)
    fingerprint = model_fingerprint(snapshot)
    report = {
        "clip_cosine_mean": paired.mean().item(),
        "clip_cosine_std": paired.std(unbiased=len(paired) > 1).item(),
        "mismatched_caption_cosine_mean": mismatched.mean().item(),
        "paired_minus_mismatched_mean": (paired - mismatched).mean().item(),
        "assessor": str(snapshot),
        "assessor_identity": fingerprint,
        "per_example_clip_cosine": paired.tolist(),
        "scope": "Independent CLIP assessor; broad semantic alignment, not an exact compositional-adherence metric",
    }
    if contrast_pairs is not None:
        comparisons = []
        for left, right in contrast_pairs:
            if not 0 <= left < len(images) or not 0 <= right < len(images) or left == right:
                raise ValueError("Contrast pairs require two distinct valid prompt indices")
            left_margin = (image_vectors[left] * (text_vectors[left] - text_vectors[right])).sum()
            right_margin = (image_vectors[right] * (text_vectors[right] - text_vectors[left])).sum()
            comparisons.append(
                {
                    "prompt_indices": [left, right],
                    "captions": [captions[left], captions[right]],
                    "correct_minus_alternative_cosine": [left_margin.item(), right_margin.item()],
                    "preferred_correct_caption": [bool(left_margin > 0), bool(right_margin > 0)],
                }
            )
        report["contrast_pairs"] = comparisons
        margins = [
            value for row in comparisons for value in row["correct_minus_alternative_cosine"]
        ]
        if not margins:
            raise ValueError("Provide at least one contrast pair")
        report["contrast_caption_preference_fraction"] = sum(value > 0 for value in margins) / len(
            margins
        )
        report["contrast_scope"] = (
            "Small controlled CLIP preference diagnostic; not a validated counting, color-binding or spatial benchmark. Ties count as failures."
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path, help="Custom held-out image-caption JSONL")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--latest", action="store_true", help="Use online rather than EMA weights")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--scale", type=float, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--matched-noise",
        action="store_true",
        help="Use the same initial latent for every custom prompt",
    )
    parser.add_argument("--split", choices=["validation", "evaluation"], default="evaluation")
    parser.add_argument(
        "--prompts", type=Path, help="JSON list of prompt strings instead of held-out COCO captions"
    )
    parser.add_argument("--clip", type=Path)
    parser.add_argument(
        "--fid", action="store_true", help="Requires >=2048 generated and reference images"
    )
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.steps) <= 0:
        raise ValueError("Evaluation counts must be positive")
    if args.fid and (args.samples < 2048 or args.prompts):
        raise ValueError("FID requires >=2048 held-out-caption images, not a custom prompt gallery")
    seed_everything(args.seed, 4)
    with args.checkpoint.open("rb") as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
        handle.seek(0)
        digest = (
            hashlib.file_digest(handle, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else None
        )
    if digest is None:
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if state.get("format") != "text_v1":
        raise ValueError("Expected a text_train checkpoint")
    cfg = OmegaConf.create(state["config"])
    if args.base_model:
        cfg.base_model = str(args.base_model)
    if args.data_root:
        cfg.data_root = str(args.data_root)
    if model_fingerprint(cfg.base_model) != state["base_identity"]:
        raise ValueError("Base codec/text/denoiser content differs from checkpoint")
    if args.base_only and cfg.mode != "lora":
        raise ValueError("The scratch model has no pretrained denoiser baseline")
    model, vae, text_encoder, tokenizer, scheduler = load_components(cfg, torch.device(args.device))
    if not args.base_only:
        load_trainable_state(model, state["model" if args.latest else "ema"])
    model.eval()
    if args.prompts:
        captions = json.loads(args.prompts.read_text())
        if (
            not isinstance(captions, list)
            or not captions
            or any(not isinstance(c, str) or not c.strip() for c in captions)
        ):
            raise ValueError("Prompts must be a nonempty JSON list of strings")
        reference, ids = None, None
    else:
        data = CaptionDataset(
            cfg.data_root, args.split, cfg.resolution, args.samples, manifest=args.manifest
        )
        captions = [row["captions"][0] for row in data.rows]
        ids = [row["image_id"] for row in data.rows]
        reference = torch.stack([data[i][0] for i in range(len(data))]) if args.fid else None
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    generated, nfes = [], []
    started = time.perf_counter()
    for start in range(0, len(captions), args.batch_size):
        values, calls = generate_text(
            model,
            vae,
            text_encoder,
            tokenizer,
            scheduler,
            cfg,
            captions[start : start + args.batch_size],
            args.steps,
            args.scale,
            args.seed if args.matched_noise else args.seed + start,
            matched_noise=args.matched_noise,
        )
        generated.append(values)
        nfes.append(calls)
        if start % 64 == 0:
            print(
                f"Generated {min(start + args.batch_size, len(captions))}/{len(captions)}",
                flush=True,
            )
    generated = torch.cat(generated)
    seconds = time.perf_counter() - started
    save_rendered_image(generated[:64], args.output / "samples.png", nrow=8)
    torch.save(generated, args.output / "samples.pt")
    save_json(args.output / "prompts.json", {"captions": captions, "image_ids": ids})
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "training_steps": state["step"],
        "base_identity": state["base_identity"],
        "base_identity_provenance": state.get("base_identity_provenance"),
        "mode": cfg.mode,
        "weights": "base" if args.base_only else "online" if args.latest else "ema",
        "seed": args.seed,
        "batch_seeds": "fixed seed with identical initial noise per prompt"
        if args.matched_noise
        else "seed + first caption index",
        "cfg_scale": args.scale,
        "steps": args.steps,
        "nfe_per_image": nfes[0],
        "nfe_scope": "Denoiser branch calls; excludes codec/text encoder",
        "generated_count": len(generated),
        "sampling_seconds": seconds,
        "resolution": cfg.resolution,
        "reference_split": args.split if ids else None,
        "training_data": "COCO 2017 train"
        if cfg.train_manifest is None
        else "Custom image-caption manifest",
        "pretraining_scope": "Pretrained SD2/VAE/text data may overlap evaluation images; fine-tuning split isolation is not proof of pretraining disjointness",
        "center_crop_scope": "Image-caption pairs use deterministic center crops; some captioned peripheral objects may be removed",
    }
    # Release GPU memory before loading independent metric networks.
    del model, vae, text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.clip:
        report.update(clip_alignment(generated, captions, args.clip, args.device))
    if args.fid:
        if len(reference) < 2048:
            raise ValueError("Insufficient reference images for requested FID")
        report.update(image_metrics(reference, generated, args.device, args.batch_size))
    save_json(args.output / "metrics.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"assessor_identity", "per_example_clip_cosine"}
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
