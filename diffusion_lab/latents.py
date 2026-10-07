"""Frozen KL autoencoders and reproducible posterior caches for latent generation.

The cache stores posterior means AND standard deviations, not a single frozen
posterior draw. Gaussian draws and augmentation choices happen in the trainer
with checkpointed RNG. Whitening statistics use training data only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from .data import ImageDataset, balanced_image_indices
from .utils import save_json, seed_everything


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_latents(z, normalization):
    mean = z.new_tensor(normalization["mean"])[None, :, None, None]
    std = z.new_tensor(normalization["std"])[None, :, None, None]
    if (std <= 0).any():
        raise ValueError("Latent whitening standard deviations must be positive")
    return (z - mean) / std


def denormalize_latents(z, normalization):
    mean = z.new_tensor(normalization["mean"])[None, :, None, None]
    std = z.new_tensor(normalization["std"])[None, :, None, None]
    return z * std + mean


def posterior_samples(moments, normalization, generator=None, augment=True):
    """Draw normalized z from q(z|image); optional variant is encoded pixel flip."""
    if moments.ndim != 5 or moments.shape[2] % 2:
        raise ValueError("Expected cached (batch,variants,2*channels,height,width) moments")
    variant = (
        torch.randint(moments.shape[1], (len(moments),), device=moments.device, generator=generator)
        if augment
        else torch.zeros(len(moments), dtype=torch.long, device=moments.device)
    )
    mean, std = (
        moments[torch.arange(len(moments), device=moments.device), variant].float().chunk(2, 1)
    )
    if (std < 0).any():
        raise ValueError("Posterior standard deviations cannot be negative")
    noise = torch.randn(mean.shape, device=mean.device, generator=generator)
    return normalize_latents(mean + std * noise, normalization)


class FrozenAutoencoder:
    """Decode with precisely the encoder scale/shift and train-only whitening.

    Requires an optional diffusers dependency and a local AutoencoderKL folder.
    The diffusion model itself is trained from scratch; no pretrained denoiser
    is loaded. Both encoder and decoder stay frozen in float32.
    """

    def __init__(self, path, device="cpu", expected=None):
        from diffusers import AutoencoderKL

        path = Path(path).expanduser().resolve()
        config_path = path / "config.json"
        weights = path / "diffusion_pytorch_model.safetensors"
        if not config_path.exists() or not weights.exists():
            raise FileNotFoundError(
                "Provide a local VAE folder containing config.json and diffusion_pytorch_model.safetensors"
            )
        metadata = {
            "path": str(path),
            "weights_sha256": file_hash(weights),
            "config_sha256": file_hash(config_path),
            "weights_file": weights.name,
        }
        if expected is not None:
            for key in ["weights_sha256", "config_sha256"]:
                if metadata[key] != expected[key]:
                    raise ValueError(f"Autoencoder {key} differs from the training cache")
        self.model = (
            AutoencoderKL.from_pretrained(
                path, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        cfg = self.model.config
        if cfg.get("latents_mean") is not None or cfg.get("latents_std") is not None:
            raise ValueError(
                "VAE-internal latents_mean/std conventions require an explicit adapter"
            )
        self.scale = float(cfg.get("scaling_factor", 1.0))
        self.shift = float(cfg.get("shift_factor") or 0.0)
        if self.scale <= 0:
            raise ValueError("VAE scaling factor must be positive")
        self.channels = cfg.latent_channels
        self.divisor = 2 ** (len(cfg.block_out_channels) - 1)
        metadata.update(
            {
                "scaling_factor": self.scale,
                "shift_factor": self.shift,
                "latent_channels": self.channels,
                "spatial_divisor": self.divisor,
                "precision": "float32",
                "pretrained_component": "frozen AutoencoderKL only",
            }
        )
        self.metadata = metadata
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_moments(self, images):
        with torch.autocast(device_type=self.device.type, enabled=False):
            posterior = self.model.encode(images.to(self.device, dtype=torch.float32)).latent_dist
        return torch.cat(
            [(posterior.mean - self.shift) * self.scale, posterior.std * self.scale], dim=1
        )

    @torch.no_grad()
    def decode(self, normalized, normalization, batch_size=8):
        if batch_size < 1:
            raise ValueError("Decoder batch size must be positive")
        output = []
        for values in normalized.split(batch_size):
            z = denormalize_latents(values.to(self.device, dtype=torch.float32), normalization)
            with torch.autocast(device_type=self.device.type, enabled=False):
                decoded = self.model.decode(z / self.scale + self.shift).sample
            if not torch.isfinite(decoded).all():
                raise FloatingPointError("Autoencoder produced nonfinite images")
            output.append(decoded.cpu())
        return torch.cat(output)


class LatentDataset(Dataset):
    def __init__(self, config, split="train"):
        self.root, self.split = Path(config.root).expanduser(), split
        path = self.root / "manifest.json"
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("format") != "latent_cache_v1":
            raise ValueError("Expected a completed latent cache manifest")
        self.manifest_sha256 = file_hash(path)
        if (config.channels, config.image_size) != (
            self.manifest["latent_channels"],
            self.manifest["latent_size"],
        ):
            raise ValueError("Data channels/image_size disagree with the latent cache")
        self.class_names = self.manifest["class_names"]
        labels = np.load(self.root / f"{split}_labels.npy", mmap_mode="r")
        if len(labels) != self.manifest["splits"][split]["count"]:
            raise ValueError("Latent labels disagree with manifest counts")
        moments_path = self.root / f"{split}_moments.npy"
        expected = (
            len(labels),
            self.manifest["splits"][split]["variants"],
            config.channels * 2,
            config.image_size,
            config.image_size,
        )
        if np.load(moments_path, mmap_mode="r").shape != expected:
            raise ValueError("Latent moment shape differs from manifest")
        if config.get("verify_cache", True):
            for filename, key in [
                (moments_path, "moments_sha256"),
                (self.root / f"{split}_labels.npy", "labels_sha256"),
            ]:
                if file_hash(filename) != self.manifest["splits"][split][key]:
                    raise ValueError("Latent cache content hash differs from manifest")
        self.files = [(index, int(label)) for index, label in enumerate(labels)]
        limit = config.train_limit if split == "train" else config.validation_limit
        if limit:
            grouped = {}
            for row in self.files:
                grouped.setdefault(row[1], []).append(row)
            self.files = [
                row
                for index in range(max(map(len, grouped.values()), default=0))
                for rows in grouped.values()
                if index < len(rows)
                for row in [rows[index]]
            ][:limit]
        self._moments = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self._moments is None:
            self._moments = np.load(self.root / f"{self.split}_moments.npy", mmap_mode="r")
        row, label = self.files[index]
        return torch.from_numpy(np.array(self._moments[row], copy=True)), label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_moments"] = None
        return state


def image_or_latent_dataset(config, split="train"):
    return LatentDataset(config, split) if config.kind == "latents" else ImageDataset(config, split)


def decode_trajectory(codec, trajectory, normalization, frames=20):
    indices = np.unique(
        np.linspace(0, len(trajectory) - 1, min(frames, len(trajectory))).astype(int)
    )
    return torch.stack([codec.decode(trajectory[index, :16], normalization) for index in indices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=["celeba", "ffhq", "cifar10"], default="celeba")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/celeba128_latent_cache"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--horizontal-flips", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    from .train import CONFIG_ROOT

    seed_everything(42, 4)
    if min(args.image_size, args.batch_size) < 1 or args.workers < 0:
        raise ValueError("Invalid encoding counts")
    if (args.output / "manifest.json").exists() or any(args.output.glob("*_moments.npy")):
        raise ValueError("Choose a new cache directory to preserve existing artifacts")
    args.output.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(CONFIG_ROOT / "data" / f"{args.data}.yaml")
    config.image_size, config.channels = args.image_size, 3
    config.train_limit, config.validation_limit = args.train_limit, args.validation_limit
    config.horizontal_flip = False
    if args.data_root:
        config.root = str(args.data_root)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    codec = FrozenAutoencoder(args.vae, args.device)
    if args.image_size % codec.divisor:
        raise ValueError("Image resolution must divide the VAE's downsampling factor")
    size, channels = args.image_size // codec.divisor, codec.channels
    manifest = {
        "format": "latent_cache_v1",
        "image_data": OmegaConf.to_container(config, resolve=True),
        "image_size": args.image_size,
        "latent_size": size,
        "latent_channels": channels,
        "autoencoder": codec.metadata,
        "splits": {},
    }
    sums, seconds, counts = np.zeros(channels), np.zeros(channels), 0
    for split in ["train", "validation"]:
        data = ImageDataset(config, split)
        if split == "train":
            manifest["class_names"] = data.class_names
        elif data.class_names != manifest["class_names"]:
            raise ValueError("Pixel train/validation classes disagree")
        variants = 2 if split == "train" and args.horizontal_flips else 1
        destination = args.output / f"{split}_moments.npy"
        moments = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=np.float16,
            shape=(len(data), variants, channels * 2, size, size),
        )
        labels = np.empty(len(data), dtype=np.int64)
        loader = DataLoader(
            data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=True,
        )
        offset = 0
        for images, classes in loader:
            encoded = torch.stack(
                [
                    codec.encode_moments(images.flip(-1) if variant else images)
                    for variant in range(variants)
                ],
                dim=1,
            )
            if not torch.isfinite(encoded).all() or encoded.abs().max() > np.finfo(np.float16).max:
                raise FloatingPointError("Posterior cache exceeds finite float16 range")
            values = encoded.cpu().numpy().astype(np.float16)
            moments[offset : offset + len(images)] = values
            labels[offset : offset + len(images)] = classes.numpy()
            if split == "train":
                m, s = (
                    values.astype(np.float64)
                    .reshape(-1, channels * 2, size, size)
                    .reshape(-1, 2, channels, size, size)
                    .transpose(1, 0, 2, 3, 4)
                )
                sums += m.sum(axis=(0, 2, 3))
                seconds += (m**2 + s**2).sum(axis=(0, 2, 3))
                counts += m.shape[0] * size**2
            offset += len(images)
            if offset % (args.batch_size * 50) == 0 or offset == len(data):
                print(
                    json.dumps({"split": split, "encoded_images": offset, "total": len(data)}),
                    flush=True,
                )
        moments.flush()
        del moments
        labels_path = args.output / f"{split}_labels.npy"
        np.save(labels_path, labels)
        source_rows = [
            (str(p.relative_to(Path(config.root))), label, p.stat().st_size)
            for p, label in data.files
        ]
        manifest["splits"][split] = {
            "count": len(data),
            "variants": variants,
            "moments_sha256": file_hash(destination),
            "labels_sha256": file_hash(labels_path),
            "source_index_sha256": hashlib.sha256(json.dumps(source_rows).encode()).hexdigest(),
        }
    mean = sums / counts
    std = np.sqrt(np.maximum(seconds / counts - mean**2, 1e-12))
    manifest["normalization"] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "estimated_from": "Training posterior mean and variance; both pixel flip variants when enabled",
    }
    save_json(args.output / "manifest.json", manifest)
    heldout = ImageDataset(config, "validation")
    indices = balanced_image_indices(heldout, min(128, len(heldout)))
    images = torch.stack([heldout[index][0] for index in indices])
    encoded = torch.cat(
        [codec.encode_moments(batch).cpu() for batch in images.split(args.batch_size)]
    )
    means = normalize_latents(encoded[:, :channels], manifest["normalization"])
    reconstructions = codec.decode(means, manifest["normalization"])
    mse = ((images.clamp(-1, 1) - reconstructions.clamp(-1, 1)) / 2).square().flatten(1).mean(1)
    save_image(
        (torch.stack([images[:32], reconstructions[:32]], dim=1).flatten(0, 1).clamp(-1, 1) + 1)
        / 2,
        args.output / "reconstruction.png",
        nrow=8,
    )
    save_json(
        args.output / "reconstruction_metrics.json",
        {
            "heldout_count": len(indices),
            "mean_psnr_db": float((-10 * mse.clamp_min(1e-12).log10()).mean()),
            "reconstruction": "Posterior mean, frozen autoencoder, clipped pixels",
            "manifest_sha256": file_hash(args.output / "manifest.json"),
            "autoencoder": codec.metadata,
        },
    )
    torch.save(
        {"reference": images, "reconstruction": reconstructions}, args.output / "reconstructions.pt"
    )
    print("Latent cache and held-out reconstruction check completed.", flush=True)


if __name__ == "__main__":
    main()
