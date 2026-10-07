"""Controlled two-dimensional GAN study with validation-only model selection.

Five distinct objective/stabilization combinations; weight clipping is retained
as a historical diagnostic. Optimizer updates and critic ratios are explicit.
Published generators use small safe NPZ files; full resume state stays private.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .data import GaussianMixture
from .metrics import gmm_metrics, mmd_squared
from .resume import capture_rng, restore_rng
from .utils import EMA, atomic_checkpoint, save_json, seed_everything
from .visualize import scatter_comparison

METHODS = (
    "non_saturating_r1",
    "least_squares",
    "hinge_spectral",
    "wasserstein_gp",
    "wasserstein_clip",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def network(output=2, spectral=False):
    layers = []
    for inputs, outputs in [(2, 128), (128, 128), (128, output)]:
        layer = nn.Linear(inputs, outputs)
        layers.append(spectral_norm(layer) if spectral else layer)
        if outputs != output or inputs != 128:
            layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def gradient_penalty(critic, real, fake, generator):
    weight = torch.rand(len(real), 1, device=real.device, generator=generator)
    mixed = (weight * real + (1 - weight) * fake).detach().requires_grad_(True)
    gradient = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=True)[0]
    return (gradient.norm(dim=1) - 1).square().mean()


def critic_objective(method, critic, real, fake, generator, update):
    real = real.detach().requires_grad_(method == "non_saturating_r1" and update % 16 == 0)
    pos, neg = critic(real), critic(fake.detach())
    if method == "least_squares":
        return 0.5 * ((pos - 1).square().mean() + neg.square().mean())
    if method == "hinge_spectral":
        return F.relu(1 - pos).mean() + F.relu(1 + neg).mean()
    if method.startswith("wasserstein"):
        loss = neg.mean() - pos.mean()
        return (
            loss + 10 * gradient_penalty(critic, real, fake, generator)
            if method.endswith("gp")
            else loss
        )
    loss = F.softplus(-pos).mean() + F.softplus(neg).mean()
    if real.requires_grad:
        gradient = torch.autograd.grad(pos.sum(), real, create_graph=True)[0]
        loss = loss + (16 * 1.0 / 2) * gradient.square().sum(1).mean()
    return loss


def generator_objective(method, values):
    if method == "least_squares":
        return 0.5 * (values - 1).square().mean()
    return F.softplus(-values).mean() if method == "non_saturating_r1" else -values.mean()


def export(generator, folder, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "generator.npz",
        **{k: v.detach().cpu().numpy() for k, v in generator.state_dict().items()},
    )
    save_json(
        folder / "generator.json",
        dict(schema=1, width=128, weights_sha256=digest(folder / "generator.npz"), **metadata),
    )


def load_generator(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "generator.json").read_text())
    if (
        metadata.get("schema") != 1
        or metadata.get("width") != 128
        or digest(folder / "generator.npz") != metadata["weights_sha256"]
    ):
        raise ValueError("Unsupported or modified GAN generator")
    model = network().eval().requires_grad_(False)
    with np.load(folder / "generator.npz", allow_pickle=False) as weights:
        if set(weights) != set(model.state_dict()):
            raise ValueError("Unexpected generator weights")
        state = {}
        for key, tensor in model.state_dict().items():
            array = weights[key]
            if (
                array.dtype != np.float32
                or array.shape != tuple(tensor.shape)
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid generator weights")
            state[key] = torch.from_numpy(array.copy())
    model.load_state_dict(state)
    return model


def train(
    method,
    seed,
    steps=6000,
    root=Path("assets/adversarial"),
    private=Path("outputs/adversarial"),
    resume=False,
):
    if method not in METHODS or steps < 1:
        raise ValueError("Unknown method or invalid update count")
    seed_everything(seed, 1)
    device = torch.device("cpu")
    folder, work = Path(root) / f"{method}_{seed}", Path(private) / f"{method}_{seed}"
    sources = {
        f: digest(Path(__file__).parent / f)
        for f in ["adversarial.py", "data.py", "metrics.py", "resume.py", "utils.py"]
    }
    config = {
        "method": method,
        "seed": seed,
        "steps": steps,
        "batch_size": 256,
        "width": 128,
        "components": 8,
        "radius": 3.0,
        "std": 0.25,
        "ema_decay": 0.995,
    }
    if (folder / "metrics.json").exists() and not resume:
        old = json.loads((folder / "metrics.json").read_text())
        if old["config"] != config or old["source_sha256"] != sources:
            raise ValueError(
                "Existing evidence belongs to a different protocol/source; use a new output"
            )
        print("Verified completed", folder, flush=True)
        return
    work.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work / "checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise ValueError("Incomplete run exists; explicitly resume or use another directory")
    gen, critic = network(), network(1, method == "hinge_spectral")
    ema = EMA(gen, 0.995)
    clipped = method == "wasserstein_clip"
    ratio = 5 if method.startswith("wasserstein") else 1

    def make_optimizer(params):
        return (
            torch.optim.RMSprop(params, lr=5e-5)
            if clipped
            else torch.optim.Adam(params, lr=1e-4, betas=(0.0, 0.9))
        )

    opt_g, opt_d = make_optimizer(gen.parameters()), make_optimizer(critic.parameters())
    mixture = GaussianMixture()
    data_rng, noise_rng, penalty_rng = [
        torch.Generator().manual_seed(seed + k) for k in [1000, 2000, 3000]
    ]
    val_z = torch.randn(1024, 2, generator=torch.Generator().manual_seed(100000))
    val_ref = mixture.sample(1024, torch.Generator().manual_seed(200000))[0]
    with torch.no_grad():
        best_score = mmd_squared(ema.model(val_z), val_ref)
    best_weights, best_step = copy.deepcopy(ema.model.state_dict()), 0
    history, first, critic_updates, cumulative_seconds = [], 0, 0, 0.0
    if resume:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["config"] != config or state["source_sha256"] != sources:
            raise ValueError("Resume configuration/source mismatch")
        for model, key in [(gen, "generator"), (critic, "critic"), (ema.model, "ema")]:
            model.load_state_dict(state[key])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        for rng, key in [
            (data_rng, "data_rng"),
            (noise_rng, "noise_rng"),
            (penalty_rng, "penalty_rng"),
        ]:
            rng.set_state(state[key])
        restore_rng(state["rng"], device)
        first, best_step, best_score, best_weights = (
            state["step"],
            state["best_step"],
            state["best_score"],
            state["best_weights"],
        )
        history, critic_updates, cumulative_seconds = (
            state["history"],
            state["critic_updates"],
            state["seconds"],
        )
    started = time.perf_counter()
    for step in range(first + 1, steps + 1):
        critic.requires_grad_(True).train()
        for _ in range(ratio):
            critic_updates += 1
            real = mixture.sample(256, data_rng)[0]
            with torch.no_grad():
                fake = gen(torch.randn(256, 2, generator=noise_rng))
            loss_d = critic_objective(method, critic, real, fake, penalty_rng, critic_updates)
            if not torch.isfinite(loss_d):
                raise FloatingPointError("Nonfinite critic objective")
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            if clipped:
                with torch.no_grad():
                    for parameter in critic.parameters():
                        parameter.clamp_(-0.01, 0.01)
        critic.requires_grad_(False)
        loss_g = generator_objective(method, critic(gen(torch.randn(256, 2, generator=noise_rng))))
        if not torch.isfinite(loss_g):
            raise FloatingPointError("Nonfinite generator objective")
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        ema.update(gen)
        if step % 500 == 0 or step == steps:
            with torch.no_grad():
                score = mmd_squared(ema.model(val_z), val_ref)
            if score < best_score:
                best_score, best_weights, best_step = (
                    score,
                    copy.deepcopy(ema.model.state_dict()),
                    step,
                )
            history.append(
                {
                    "step": step,
                    "validation_mmd": score,
                    "generator_loss": float(loss_g.detach()),
                    "critic_loss": float(loss_d.detach()),
                }
            )
            atomic_checkpoint(
                checkpoint_path,
                {
                    "config": config,
                    "source_sha256": sources,
                    "generator": gen.state_dict(),
                    "critic": critic.state_dict(),
                    "ema": ema.model.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "data_rng": data_rng.get_state(),
                    "noise_rng": noise_rng.get_state(),
                    "penalty_rng": penalty_rng.get_state(),
                    "rng": capture_rng(device),
                    "step": step,
                    "best_step": best_step,
                    "best_score": best_score,
                    "best_weights": best_weights,
                    "history": history,
                    "critic_updates": critic_updates,
                    "seconds": cumulative_seconds + time.perf_counter() - started,
                },
            )
            print(method, seed, step, score, "best", best_step, flush=True)
    selected = network().eval().requires_grad_(False)
    selected.load_state_dict(best_weights)
    z = torch.randn(4096, 2, generator=torch.Generator().manual_seed(300000))
    reference = mixture.sample(4096, torch.Generator().manual_seed(400000))[0]
    with torch.no_grad():
        samples = selected(z)
    metric = gmm_metrics(samples, reference, mixture)
    floor = gmm_metrics(
        mixture.sample(4096, torch.Generator().manual_seed(500000))[0], reference, mixture
    )
    export(selected, folder, {"method": method, "seed": seed, "selected_step": best_step})
    with torch.no_grad():
        portable = load_generator(folder)(z)
    torch.testing.assert_close(portable, samples, rtol=0, atol=0)
    np.savez_compressed(
        folder / "samples.npz",
        latent=z.numpy(),
        generated=samples.numpy(),
        reference=reference.numpy(),
    )
    scatter_comparison(
        reference,
        samples,
        folder / "samples.png",
        f"{method}; seed {seed}, selected update {best_step}",
    )
    save_json(folder / "history.json", history)
    metric.update(
        config=config,
        source_sha256=sources,
        selected_step=best_step,
        validation_mmd=best_score,
        critic_updates=critic_updates,
        generator_updates=steps,
        training_seconds=cumulative_seconds + time.perf_counter() - started,
        reference_floor=floor,
        samples_sha256=digest(folder / "samples.npz"),
        generator_sha256=digest(folder / "generator.npz"),
        nfe_per_sample=1,
        selection="First minimum EMA validation MMD, 1024 independent fixed draws, every 500 updates including initial/final; test 4096 fixed independent draws only after selection",
        scope="Synthetic 8-component GMM. Critic ratios/optimizer and training budgets differ from historical diffusion/flow; target cross entropy is not generator NLL. No image benchmark or SOTA claim.",
    )
    save_json(folder / "metrics.json", metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--output", type=Path, default=Path("assets/adversarial"))
    parser.add_argument("--private", type=Path, default=Path("outputs/adversarial"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for method in args.method or METHODS:
        train(method, args.seed, args.steps, args.output, args.private, args.resume)


if __name__ == "__main__":
    main()
