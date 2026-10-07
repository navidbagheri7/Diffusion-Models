"""Evaluate latent generators in decoded image space, retaining representation provenance."""

import time

import torch
from omegaconf import OmegaConf

from .data import ImageDataset, balanced_image_indices
from .latents import FrozenAutoencoder, decode_trajectory, normalize_latents
from .metrics import image_metrics, image_pixels
from .utils import atomic_checkpoint, save_json
from .visualize import image_artifacts, save_rendered_image


def evaluate_latents(args, cfg, model, classes, training_step):
    from .image_evaluate import slerp
    from .image_train import sample_images

    if args.classifier or args.clip_snapshot or args.feature_classifier:
        raise ValueError(
            "Pixel noisy classifiers are not latent noisy classifiers; no adapter is assumed"
        )
    if args.real_interpolation or args.matched_class_noise:
        raise ValueError(
            "This latent lesson supports generated latent interpolation and cached class galleries"
        )
    if args.class_label is not None and not 0 <= args.class_label < classes:
        raise ValueError("Class label outside latent model conditioning")
    device = next(model.parameters()).device
    manifest = model.latent_manifest
    codec = FrozenAutoencoder(
        args.vae or manifest["autoencoder"]["path"], device, manifest["autoencoder"]
    )
    normalization = manifest["normalization"]
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_generator = torch.Generator(device=device).manual_seed(args.seed)
    decoded, latent_outputs, labels_saved, calls, sizes, history = [], [], [], [], [], None
    measured_calls = [0]

    def count_forward(_module, _inputs, _output):
        measured_calls[0] += 1

    handle = model.register_forward_hook(count_forward)
    started = time.perf_counter()
    for offset in range(0, args.samples, args.batch_size):
        size = min(args.batch_size, args.samples - offset)
        noise = torch.randn(
            size,
            cfg.data.channels,
            cfg.data.image_size,
            cfg.data.image_size,
            generator=initial_generator,
            device=device,
        )
        labels = (
            torch.full((size,), args.class_label, device=device, dtype=torch.long)
            if args.class_label is not None
            else torch.arange(offset, offset + size, device=device) % classes
            if classes
            else None
        )
        before = measured_calls[0]
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"
        ):
            latent, trajectory, nfe = sample_images(
                cfg, model, noise, labels, args.scale, args.method, args.steps, generator
            )
        decoded.append(codec.decode(latent, normalization))
        latent_outputs.append(latent.cpu())
        if labels is not None:
            labels_saved.append(labels.cpu())
        actual_calls = measured_calls[0] - before
        if actual_calls != nfe:
            raise RuntimeError("Sampler NFE disagrees with measured denoiser forward calls")
        calls.append(actual_calls)
        sizes.append(size)
        if history is None:
            history = decode_trajectory(codec, trajectory, normalization)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    images, latents = torch.cat(decoded), torch.cat(latent_outputs)
    labels = torch.cat(labels_saved) if labels_saved else None
    image_artifacts(images, history, args.output)
    if labels is not None:
        shown = [
            images[labels == label][:8]
            for label in range(min(classes, 10))
            if (labels == label).any()
        ]
        save_rendered_image(torch.cat(shown), args.output / "classes.png", nrow=8)
    pixel_config = OmegaConf.create(manifest["image_data"])
    if args.data_root:
        pixel_config.root = str(args.data_root)
    pixel_config.horizontal_flip = False
    pixel_config.train_limit = pixel_config.validation_limit = 0
    data = ImageDataset(pixel_config, args.reference_split)
    selected = balanced_image_indices(data, args.samples, args.class_label)
    if not selected:
        raise ValueError("No reference images for this latent class")
    reference = torch.stack([data[index][0] for index in selected])
    metrics = (
        {"inception_metrics": "disabled by argument", "real_count": len(reference)}
        if args.skip_metrics
        else image_metrics(reference, images, device, args.batch_size)
    )
    if args.reconstruction_baseline:
        encoded = torch.cat(
            [
                codec.encode_moments(batch).cpu()[:, : cfg.data.channels]
                for batch in reference.split(8)
            ]
        )
        reconstruction = codec.decode(normalize_latents(encoded, normalization), normalization)
        save_rendered_image(reconstruction[:64], args.output / "reconstructions.png", nrow=8)
        baseline = (
            image_metrics(reference, reconstruction, device, args.batch_size)
            if not args.skip_metrics
            else {}
        )
        baseline["psnr_db"] = float(
            (
                -10
                * (
                    ((reference.clamp(-1, 1) - reconstruction.clamp(-1, 1)) / 2)
                    .square()
                    .flatten(1)
                    .mean(1)
                )
                .clamp_min(1e-12)
                .log10()
            ).mean()
        )
        baseline["kind"] = (
            "Frozen VAE posterior-mean reconstructions of the same held-out reference"
        )
        save_json(args.output / "reconstruction_metrics.json", baseline)
    if args.interpolation:
        pair = torch.randn(
            2,
            cfg.data.channels,
            cfg.data.image_size,
            cfg.data.image_size,
            device=device,
            generator=generator,
        )
        interpolated = slerp(pair[0], pair[1], torch.linspace(0, 1, 9, device=device))
        conditions = (
            torch.full((9,), args.class_label or 0, dtype=torch.long, device=device)
            if classes
            else None
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"
        ):
            values, _, _ = sample_images(
                cfg, model, interpolated, conditions, args.scale, args.method, args.steps, generator
            )
        interpolation = codec.decode(values, normalization).clamp(-1, 1)
        save_rendered_image(
            interpolation,
            args.output / "interpolation.png",
            nrow=9,
        )
        from PIL import Image

        frames = [
            Image.fromarray(image_pixels(frame[None])[0].permute(1, 2, 0).numpy())
            for frame in interpolation
        ]
        frames[0].save(
            args.output / "interpolation.gif",
            save_all=True,
            append_images=frames[1:] + frames[-2:0:-1],
            duration=160,
            loop=0,
        )
    metrics.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": model.checkpoint_sha256,
            "training_steps": training_step,
            "algorithm": cfg.algorithm,
            "backbone": cfg.model.kind,
            "representation": "Whitened autoencoder posterior; metrics use decoded RGB images",
            "image_resolution": manifest["image_size"],
            "latent_shape": list(latents.shape[1:]),
            "autoencoder": codec.metadata,
            "normalization": normalization,
            "seed": args.seed,
            "generated_count": len(images),
            "reference_split": args.reference_split,
            "nfe_per_image": sum(c * n for c, n in zip(calls, sizes, strict=True)) / len(images),
            "nfe_per_image_range": [min(calls), max(calls)],
            "decoder_calls_excluded_from_nfe": True,
            "sampling_seconds_includes_decoding_and_first_trajectory": seconds,
            "cfg_scale": args.scale,
            "class_label": args.class_label,
            "sampler": cfg.algorithm
            if cfg.algorithm
            in {"consistency", "distillation", "improved_consistency", "progressive"}
            else args.method or "heun",
            "steps": args.steps or cfg.sampling_steps,
            "precision": "bfloat16 denoiser/float32 autoencoder"
            if cfg.amp and device.type == "cuda"
            else "float32",
            "batch_size": args.batch_size,
        }
    )
    atomic_checkpoint(
        args.output / "samples.pt",
        {
            "samples": images,
            "latents": latents,
            "labels": labels,
            "reference": reference,
            "reference_labels": torch.tensor([data.files[i][1] for i in selected]),
            "trajectory": history,
        },
    )
    save_json(args.output / "metrics.json", metrics)
    handle.remove()
    print(metrics, flush=True)
