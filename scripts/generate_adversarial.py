"""Generate 2D samples with a small public, digest-checked GAN model on CPU."""

import argparse
from pathlib import Path

import numpy as np
import torch

from diffusion_lab.adversarial import load_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=300000)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.output.exists():
        raise FileExistsError("Choose a new output file to preserve earlier samples")
    torch.set_num_threads(1)
    model = load_generator(args.model)
    latent = torch.randn(args.samples, 2, generator=torch.Generator().manual_seed(args.seed))
    with torch.no_grad():
        generated = model(latent).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, latent=latent.numpy(), generated=generated)
    print("Generated", args.samples, "2D samples on CPU:", args.output)


if __name__ == "__main__":
    main()
