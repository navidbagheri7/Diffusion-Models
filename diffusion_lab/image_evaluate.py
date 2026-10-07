"""Evaluate image_v2 checkpoints, CFG, samplers, metrics and interpolation."""

import argparse
import time
from itertools import pairwise
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .data import ImageDataset, balanced_image_indices
from .edm import karras_sigmas
from .image_train import load_image_run, sample_images
from .metrics import image_metrics, image_pixels
from .utils import save_json, seed_everything
from .visualize import image_artifacts, save_rendered_image


def slerp(first, second, fractions):
    """Spherical interpolation of directions with linearly interpolated radii."""
    shape = first.shape
    a, b = first.flatten(), second.flatten()
    na, nb = a.norm(), b.norm()
    cosine = (a @ b / (na * nb).clamp_min(1e-12)).clamp(-1, 1)
    angle = cosine.acos()
    outputs = []
    for fraction in fractions:
        if angle.sin().abs() < 1e-5:
            value = (1 - fraction) * a + fraction * b
        else:
            direction = ((1 - fraction) * angle).sin() / angle.sin() * a / na + (
                fraction * angle
            ).sin() / angle.sin() * b / nb
            value = ((1 - fraction) * na + fraction * nb) * direction
        outputs.append(value.reshape(shape))
    return torch.stack(outputs)


@torch.no_grad()
def invert_edm(model, clean, sigma_min, sigma_max, steps=100, labels=None):
    """Deterministic approximate PF-ODE inversion; endpoints are not exact recovery."""
    grid = karras_sigmas(steps, sigma_min, sigma_max, device=clean.device)[:-1].flip(0)
    x = clean
    for low, high in pairwise(grid):
        time = low.expand(len(x))
        drift = (x - model(x, time, labels)) / low
        proposal = x + (high - low) * drift
        second = (proposal - model(proposal, high.expand(len(x)), labels)) / high
        x = x + (high - low) * (drift + second) / 2
    return x / sigma_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--clip-consistency",
        action="store_true",
        help="Clip each consistency prediction to [-1,1] before re-noising",
    )
    parser.add_argument(
        "--consistency-sigmas",
        type=float,
        nargs="+",
        help="Explicit descending consistency evaluation levels, including sigma_max",
    )
    parser.add_argument("--method")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--class-label", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--matched-class-noise", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--real-interpolation", action="store_true")
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--feature-classifier", type=Path)
    parser.add_argument("--clip-snapshot", type=Path)
    parser.add_argument("--clip-prompts", type=Path)
    parser.add_argument("--classifier-scale", type=float, default=1.0)
    parser.add_argument("--classifier-sigma-min", type=float)
    parser.add_argument("--classifier-sigma-max", type=float)
    parser.add_argument("--reference-split", choices=["train", "validation"], default="validation")
    parser.add_argument("--reference-offset", type=int, default=0)
    parser.add_argument(
        "--vae", type=Path, help="Relocated latent autoencoder folder; hashes must match"
    )
    parser.add_argument(
        "--data-root", type=Path, help="Relocated original image data for latent evaluation"
    )
    parser.add_argument("--reconstruction-baseline", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.batch_size) <= 0:
        raise ValueError("Sampling counts must be positive")
    if args.reference_offset < 0:
        raise ValueError("Reference offset must be nonnegative")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("Sampling steps must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    cfg, model, classes, training_step = load_image_run(args.checkpoint, device)
    if args.clip_consistency and (
        cfg.algorithm not in {"consistency", "distillation", "improved_consistency"}
        or cfg.data.kind == "latents"
    ):
        raise ValueError(
            "Clipped consistency refinement is defined only for pixel consistency models"
        )
    if args.consistency_sigmas is not None:
        if cfg.algorithm not in {"consistency", "distillation", "improved_consistency"}:
            raise ValueError("Explicit consistency levels require a consistency model")
        if args.steps is not None and args.steps != len(args.consistency_sigmas):
            raise ValueError("steps must match the explicit consistency level count")
        args.steps = len(args.consistency_sigmas)
    if cfg.data.kind == "latents":
        from .latent_evaluate import evaluate_latents

        return evaluate_latents(args, cfg, model, classes, training_step)
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    classifier = None
    if args.classifier and args.clip_snapshot:
        raise ValueError("Choose a trained classifier or CLIP")
    if args.classifier:
        from .image_guidance import load_classifier

        classifier = load_classifier(args.classifier, device)
        if classifier.assessment_only:
            raise ValueError("A clean-only assessment classifier is not trained for noisy guidance")
        if cfg.algorithm != "edm":
            raise ValueError("The image_v2 classifier experiment currently uses EDM sigma")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_generator = torch.Generator(device=device).manual_seed(args.seed)
    if args.clip_snapshot:
        from .image_guidance import DenoisedCLIPClassifier

        if cfg.algorithm != "edm" or args.class_label is None or args.clip_prompts is None:
            raise ValueError("CLIP requires EDM, a class-label and a prompts text file")
        prompts = args.clip_prompts.read_text().splitlines()
        if len(prompts) < 2 or args.class_label >= len(prompts):
            raise ValueError("CLIP needs competing prompts and a valid prompt index")
        classifier = DenoisedCLIPClassifier(model, args.clip_snapshot, prompts, device)
    label_classes = (
        len(prompts)
        if args.clip_snapshot
        else classifier.output.out_features
        if classifier is not None
        else classes
    )
    if args.class_label is not None and not 0 <= args.class_label < label_classes:
        raise ValueError("Class label is outside the selected conditioning/guidance model")
    if classifier is not None and classes == 0 and args.class_label is None:
        raise ValueError("Classifier guidance on an unconditional model requires --class-label")
    if classifier is not None and args.real_interpolation:
        raise ValueError(
            "The real-image inversion lesson currently supports the generator ODE without classifier guidance"
        )
    if classifier is not None:
        if args.classifier_sigma_min is not None:
            classifier.guidance_sigma_min = args.classifier_sigma_min
        if args.classifier_sigma_max is not None:
            classifier.guidance_sigma_max = args.classifier_sigma_max
        if not 0 < classifier.guidance_sigma_min < classifier.guidance_sigma_max:
            raise ValueError("Expected positive ordered classifier guidance noise bounds")
    outputs, all_labels, first_history, nfes, batch_sizes = [], [], None, [], []
    measured_calls = [0]
    measured_classifier_calls = [0]
    classifier_nfes = []

    def count_classifier(_module, _inputs, _output):
        measured_classifier_calls[0] += 1

    classifier_handle = (
        classifier.register_forward_hook(count_classifier) if classifier is not None else None
    )

    def count_call(_module, _inputs, _output):
        measured_calls[0] += 1

    handle = (model.network if hasattr(model, "network") else model).register_forward_hook(
        count_call
    )
    started = time.perf_counter()
    for offset in range(0, args.samples, args.batch_size):
        before = measured_calls[0]
        before_classifier = measured_classifier_calls[0]
        size = min(args.batch_size, args.samples - offset)
        batch_sizes.append(size)
        noise = torch.randn(
            size,
            cfg.data.channels,
            cfg.data.image_size,
            cfg.data.image_size,
            device=device,
            generator=initial_generator,
        )
        labels = (
            torch.full((size,), args.class_label, device=device, dtype=torch.long)
            if args.class_label is not None
            else (torch.arange(offset, offset + size, device=device) % classes if classes else None)
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"
        ):
            generated, history, _nfe = sample_images(
                cfg,
                model,
                noise,
                labels,
                0 if classifier else args.scale,
                args.method,
                args.steps,
                generator,
                classifier,
                args.classifier_scale,
                consistency_sigmas=args.consistency_sigmas,
                clip_consistency=args.clip_consistency,
            )
        outputs.append(generated.cpu())
        if labels is not None:
            all_labels.append(labels.cpu())
        nfes.append(measured_calls[0] - before)
        classifier_nfes.append(measured_classifier_calls[0] - before_classifier)
        if first_history is None:
            first_history = history
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    generated = torch.cat(outputs)
    image_artifacts(generated, first_history, args.output)
    reference_cfg = cfg.data.copy()
    reference_cfg.horizontal_flip = False
    data = ImageDataset(reference_cfg, args.reference_split)
    selected = balanced_image_indices(data, args.samples + args.reference_offset, args.class_label)[
        args.reference_offset :
    ]
    if not selected:
        raise ValueError("No reference images of the requested class")
    reference = torch.stack([data[i][0] for i in selected[: args.samples]])
    if cfg.data.channels == 1:
        # ImageNet Inception FID/IS are not meaningful MNIST quality indicators.
        metrics = {
            "inception_metrics": "not reported for MNIST/grayscale; use digit classifier features",
            "reference_count": len(reference),
        }
    elif args.skip_metrics:
        metrics = {"inception_metrics": "disabled by argument"}
    else:
        metrics = image_metrics(reference, generated, device, args.batch_size)
    if classes and classifier is None and not args.matched_class_noise:
        labels = (
            torch.full((len(generated),), args.class_label, dtype=torch.long)
            if args.class_label is not None
            else torch.arange(len(generated)) % classes
        )
        shown = [
            generated[labels == label][:8]
            for label in range(min(classes, 10))
            if (labels == label).any()
        ]
        save_rendered_image(torch.cat(shown), args.output / "classes.png", nrow=8)
    if classes and classifier is None and args.matched_class_noise:
        noise = torch.randn(
            8,
            cfg.data.channels,
            cfg.data.image_size,
            cfg.data.image_size,
            device=device,
            generator=generator,
        )
        rows = []
        for label in range(min(classes, 10)):
            labels = torch.full((8,), label, device=device, dtype=torch.long)
            values, _, _ = sample_images(
                cfg,
                model,
                noise.clone(),
                labels,
                args.scale,
                args.method,
                args.steps,
                generator,
                consistency_sigmas=args.consistency_sigmas,
                clip_consistency=args.clip_consistency,
            )
            rows.append(values.cpu())
        save_rendered_image(torch.cat(rows), args.output / "classes.png", nrow=8)
    if cfg.data.channels == 1 and (args.feature_classifier or args.classifier):
        from .image_guidance import digit_feature_metrics, load_classifier

        features = load_classifier(args.feature_classifier or args.classifier, device)
        labels = (
            torch.full((len(generated),), args.class_label, dtype=torch.long)
            if args.class_label is not None
            else torch.arange(len(generated)) % classes
            if classes
            else None
        )
        metrics.update(digit_feature_metrics(features, reference, generated, labels, device))
    if cfg.data.channels == 3 and (args.feature_classifier or args.classifier):
        from .image_guidance import load_classifier

        assessment = load_classifier(args.feature_classifier or args.classifier, device)
        guesses = []
        for batch in generated.split(args.batch_size):
            with torch.no_grad():
                assessment_time = torch.full((len(batch),), -6.907755278982137 / 4, device=device)
                pixels = image_pixels(batch).float().div(127.5).sub(1).to(device)
                guesses.append(assessment(pixels, assessment_time).argmax(1).cpu())
        requested = torch.cat(all_labels) if all_labels else None
        if requested is not None:
            metrics["requested_class_accuracy"] = (
                (torch.cat(guesses) == requested).float().mean().item()
            )
        metrics["assessment_classifier_sha256"] = assessment.checkpoint_sha256
        metrics["assessment_classifier_sigma"] = 0.001
        metrics["assessment_preprocessing"] = (
            "Clipped and rounded uint8 images, normalized to [-1,1]"
        )
    if args.interpolation or args.real_interpolation:
        latents = torch.randn(
            2,
            cfg.data.channels,
            cfg.data.image_size,
            cfg.data.image_size,
            device=device,
            generator=generator,
        )
        if args.real_interpolation:
            if cfg.algorithm != "edm":
                raise ValueError("Real-image inversion example currently requires EDM")
            selected_clean = reference[:2].to(device)
            inversion_labels = (
                torch.full((2,), args.class_label, device=device, dtype=torch.long)
                if args.class_label is not None
                else None
            )
            latents = invert_edm(
                model, selected_clean, cfg.sigma_min, cfg.sigma_max, labels=inversion_labels
            )
            save_rendered_image(
                selected_clean.cpu(), args.output / "interpolation_inputs.png", nrow=2
            )
        interpolated = slerp(latents[0], latents[1], torch.linspace(0, 1, 9, device=device))
        labels = (
            torch.full((9,), args.class_label, device=device, dtype=torch.long)
            if args.class_label is not None
            else None
        )
        images, _, _ = sample_images(
            cfg,
            model,
            interpolated,
            labels,
            0 if classifier is not None else args.scale,
            args.method,
            args.steps,
            generator,
            classifier,
            args.classifier_scale,
            consistency_sigmas=args.consistency_sigmas,
            clip_consistency=args.clip_consistency,
        )
        save_rendered_image(images.cpu(), args.output / "interpolation.png", nrow=9)
        from PIL import Image
        from torchvision.utils import make_grid

        frames = [
            Image.fromarray(
                (make_grid((x.cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy() * 255)
                .round()
                .astype("uint8")
            ).resize((256, 256))
            for x in images
        ]
        frames[0].save(
            args.output / "interpolation.gif",
            save_all=True,
            append_images=frames[1:] + frames[-2:0:-1],
            duration=160,
            loop=0,
        )
    effective_sampler = (
        cfg.algorithm
        if cfg.algorithm in {"consistency", "distillation", "improved_consistency", "progressive"}
        else args.method or ("heun" if cfg.algorithm in {"edm", "flow"} else "dpmpp_2m")
    )
    average_nfe = sum(calls * size for calls, size in zip(nfes, batch_sizes, strict=True)) / len(
        generated
    )
    metrics.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": model.checkpoint_sha256,
            "training_steps": training_step,
            "seed": args.seed,
            "algorithm": cfg.algorithm,
            "sampler": effective_sampler,
            "steps": args.steps or cfg.sampling_steps,
            "consistency_sigmas": args.consistency_sigmas,
            "clip_consistency": args.clip_consistency,
            "nfe_per_image": nfes[0] if min(nfes) == max(nfes) else average_nfe,
            "nfe_per_image_range": [min(nfes), max(nfes)],
            "network_calls_include_cfg": True,
            "classifier_calls_per_image": classifier_nfes[0],
            "guidance_kind": "denoised_clip_heuristic"
            if args.clip_snapshot
            else "noisy_classifier"
            if args.classifier
            else "cfg"
            if classes
            else "unconditional",
            "cfg_scale": 0 if classifier is not None else args.scale,
            "classifier_scale": args.classifier_scale if classifier is not None else None,
            "class_label": args.class_label,
            "generated_count": len(generated),
            "reference_split": args.reference_split,
            "reference_offset": args.reference_offset,
            "sampling_seconds": seconds,
            "precision": "bfloat16" if cfg.amp and device.type == "cuda" else "float32",
            "batch_size": args.batch_size,
        }
    )
    if args.classifier:
        metrics["guidance_classifier_sha256"] = classifier.checkpoint_sha256
    if classifier is not None:
        metrics["guidance_sigma_range"] = [
            classifier.guidance_sigma_min,
            classifier.guidance_sigma_max,
        ]
    if args.clip_snapshot:
        metrics["clip_snapshot"] = str(args.clip_snapshot.resolve())
        metrics["clip_prompts"] = prompts
    save_json(args.output / "metrics.json", metrics)
    handle.remove()
    if classifier_handle is not None:
        classifier_handle.remove()
    torch.save(
        {
            "samples": generated,
            "labels": torch.cat(all_labels) if all_labels else None,
            "reference": reference,
            "reference_labels": torch.tensor([data.files[i][1] for i in selected]),
            "trajectory": first_history,
        },
        args.output / "samples.pt",
    )
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
