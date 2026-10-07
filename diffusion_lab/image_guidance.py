"""Noisy image classifiers and differentiable denoised-image CLIP energies."""

import argparse
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ImageDataset
from .metrics import image_pixels
from .models import Conditioning, normalization
from .utils import atomic_checkpoint, save_json, seed_everything


class NoisyImageClassifier(nn.Module):
    def __init__(self, channels=3, classes=10, width=32):
        super().__init__()
        # Avoid extrapolating a finite-data classifier into nearly pure noise.
        self.guidance_sigma_min, self.guidance_sigma_max = 0.002, 2.0
        self.condition = Conditioning(width * 4, 0)
        sizes = [width, width * 2, width * 4, width * 4]
        incoming = [channels, *sizes[:-1]]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(a, b, 3, stride=1 if i == 0 else 2, padding=1)
                for i, (a, b) in enumerate(zip(incoming, sizes, strict=True))
            ]
        )
        self.norms = nn.ModuleList([normalization(c) for c in sizes])
        self.modulations = nn.ModuleList([nn.Linear(width * 4, c * 2) for c in sizes])
        self.output = nn.Linear(width * 4, classes)

    def features(self, x, time):
        sigma = (4 * time).exp()
        h = x / (sigma[:, None, None, None].square() + 0.25).sqrt()
        embedding = self.condition(time)
        for conv, norm, modulation in zip(self.convs, self.norms, self.modulations, strict=True):
            scale, shift = modulation(embedding).chunk(2, 1)
            h = F.silu(norm(conv(h)) * (1 + scale[:, :, None, None]) + shift[:, :, None, None])
        return h.mean((2, 3))

    def forward(self, x, time):
        return self.output(self.features(x, time))


def load_classifier(path, device):
    with Path(path).open("rb") as handle:
        state = torch.load(handle, map_location=device, weights_only=False)
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    model = NoisyImageClassifier(**state["model_config"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(state["model"])
    model.checkpoint_sha256 = digest.hexdigest()
    model.assessment_only = state.get("clean_only", False)
    return model


class DenoisedCLIPClassifier(nn.Module):
    """A heuristic CLIP energy; not a trained noisy posterior classifier."""

    def __init__(self, denoiser, snapshot, prompts, device):
        super().__init__()
        self.guidance_sigma_min, self.guidance_sigma_max = 0.05, 5.0
        from transformers import CLIPModel, CLIPTokenizer

        self.denoiser = denoiser
        self.clip = (
            CLIPModel.from_pretrained(str(snapshot), local_files_only=True)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        tokenizer = CLIPTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text = self.clip.get_text_features(**inputs)
            if not isinstance(text, torch.Tensor):
                text = text.pooler_output
        self.register_buffer("text", F.normalize(text, dim=-1))
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None],
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None],
        )
        self.resolution = self.clip.config.vision_config.image_size

    def forward(self, x, time):
        clean = self.denoiser(x, (4 * time).exp(), None)
        if clean.shape[1] == 1:
            clean = clean.repeat(1, 3, 1, 1)
        pixels = F.interpolate(
            (clean.clamp(-1, 1) + 1) / 2,
            size=(self.resolution, self.resolution),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        features = self.clip.get_image_features(pixel_values=(pixels - self.mean) / self.std)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return (
            self.clip.logit_scale.exp().clamp(max=100) * F.normalize(features, dim=-1) @ self.text.T
        )


def digit_feature_metrics(classifier, real, generated, requested_labels=None, device="cpu"):
    """Domain-specific distance: explicitly not ImageNet Inception FID."""
    import numpy as np
    from scipy.linalg import sqrtm

    if min(len(real), len(generated)) < 2:
        raise ValueError("A feature covariance comparison requires two images in each set")

    features, predictions = [], None
    for images in [real, generated]:
        rows, guesses = [], []
        for batch in images.split(128):
            batch = image_pixels(batch).float().div(127.5).sub(1).to(device)
            time = torch.full((len(batch),), math.log(0.001) / 4, device=device)
            with torch.no_grad():
                value = classifier.features(batch, time)
                guesses.append(classifier.output(value).argmax(1).cpu())
                rows.append(value.cpu())
        features.append(torch.cat(rows).double().numpy())
        predictions = torch.cat(guesses)
    first, second = features
    ca, cb = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    distance = np.square(first.mean(0) - second.mean(0)).sum() + np.trace(
        ca + cb - 2 * sqrtm(ca @ cb).real
    )
    result = {
        f"frechet_digit_features_{first.shape[1]}": float(max(distance, 0)),
        "feature_dimension": first.shape[1],
        "feature_model": "locally trained digit classifier; not Inception FID",
        "feature_classifier_sha256": getattr(classifier, "checkpoint_sha256", None),
        "feature_sigma": 0.001,
        "feature_preprocessing": "Clipped and rounded uint8 images, then normalized to [-1,1]",
        "real_count": len(real),
        "generated_count": len(generated),
    }
    if requested_labels is not None:
        result["requested_class_accuracy"] = (
            (predictions == requested_labels.cpu()).float().mean().item()
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="Train a sigma-conditioned image classifier")
    parser.add_argument("--data", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--root", type=Path, default=Path("/data/tii/Image_data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/mnist_classifier"))
    args = parser.parse_args()
    if min(args.steps, args.width, args.batch_size) <= 0:
        raise ValueError("Classifier training counts and width must be positive")
    from omegaconf import OmegaConf

    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create(
        {
            "root": str(args.root / args.data),
            "channels": 1 if args.data == "mnist" else 3,
            "image_size": 32,
            "holdout_fraction": 0,
            "split_seed": 123,
            "train_limit": 0,
            "validation_limit": 0,
            "horizontal_flip": args.data == "cifar10",
        }
    )
    data, validation = ImageDataset(cfg), ImageDataset(cfg, "validation")
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    iterator = iter(loader)
    device = torch.device(args.device)
    model_config = {"channels": cfg.channels, "classes": len(data.class_names), "width": args.width}
    model = NoisyImageClassifier(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    history = []
    for step in range(args.steps):
        try:
            clean, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            clean, labels = next(iterator)
        clean, labels = clean.to(device), labels.to(device)
        sigma = (torch.randn(len(clean), device=device) * 1.5 - 1.5).exp()
        sigma[: len(clean) if args.clean_only else len(clean) // 2] = 0.001
        noisy = clean + sigma[:, None, None, None] * torch.randn_like(clean)
        loss = F.cross_entropy(model(noisy, sigma.log() / 4), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0 or step == args.steps - 1:
            history.append({"step": step + 1, "loss": loss.item()})
            atomic_checkpoint(
                args.output / "classifier.pt",
                {
                    "model": model.state_dict(),
                    "model_config": model_config,
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "data": args.data,
                    "data_config": dict(cfg),
                    "seed": args.seed,
                    "clean_only": args.clean_only,
                },
            )
            print(history[-1], flush=True)
    model.eval()
    rows = []
    heldout = DataLoader(validation, batch_size=128, shuffle=False, num_workers=2)
    for sigma in [0.001] if args.clean_only else [0.001, 0.1, 0.5, 1.0, 2.0, 5.0]:
        correct, count = 0, 0
        generator = torch.Generator(device=device).manual_seed(123)
        for clean, labels in heldout:
            clean, labels = clean.to(device), labels.to(device)
            noisy = clean + sigma * torch.randn(clean.shape, device=device, generator=generator)
            with torch.no_grad():
                predicted = model(
                    noisy, torch.full((len(clean),), sigma, device=device).log() / 4
                ).argmax(1)
            correct += (predicted == labels).sum().item()
            count += len(clean)
        rows.append({"sigma": sigma, "heldout_accuracy": correct / count, "count": count})
    save_json(args.output / "diagnostics.json", rows)
    save_json(args.output / "history.json", history)
    save_json(
        args.output / "config.json",
        {
            "model": model_config,
            "data": dict(cfg),
            "seed": args.seed,
            "clean_only": args.clean_only,
            "steps": args.steps,
            "batch_size": args.batch_size,
        },
    )


if __name__ == "__main__":
    main()
