"""Alternating neural Diffusion Schrödinger Bridge Matching (Brownian reference).

Implements the backward/forward Markovian projections and endpoint replay in
Algorithm 1 of Shi et al. (2023). Independent initialization is DSBM-IMF; reference
initialization is DSBM-IPF. Finite training/simulation only approximate the limit.
"""

import argparse
from pathlib import Path

import torch

from .bridges import bridge_training_pair
from .data import GaussianMixture
from .models import TimeMLP
from .solvers import integrate
from .toy_data import ToyDistribution, toy_metrics
from .utils import EMA, atomic_checkpoint, save_json, seed_everything
from .visualize import particle_animation, scatter_comparison


def projection_target(source, target, t, epsilon, direction, noise=None):
    """Reverse direction uses its own increasing clock r=1-t."""
    if direction == "backward":
        source, target = target, source
    elif direction != "forward":
        raise ValueError("direction must be forward or backward")
    x, drift, _ = bridge_training_pair(source, target, t, epsilon, noise)
    return x, drift


@torch.no_grad()
def simulate_projection(network, starting, epsilon, steps=200, time_eps=0.001, generator=None):
    # Start at the actual endpoint. Training excludes endpoints; the neural field
    # provides an extrapolation over the first tiny interval. Stop before singularity.
    times = torch.linspace(0, 1 - time_eps, steps + 1, device=starting.device)
    return integrate(
        lambda x, t: network(x, t),
        starting,
        times,
        "euler_maruyama",
        lambda t: torch.full_like(t, epsilon**0.5),
        generator,
    )


def fit_projection(
    network, source, target, epsilon, direction, steps, batch_size=256, lr=0.0003, time_eps=0.001
):
    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    ema = EMA(network, 0.995)
    losses = []
    for step in range(steps):
        index = torch.randint(len(source), (batch_size,), device=source.device)
        t = time_eps + (1 - 2 * time_eps) * torch.rand(batch_size, device=source.device)
        x, drift = projection_target(source[index], target[index], t, epsilon, direction)
        estimate = network(x, t)
        # Positive time weighting keeps the same population conditional expectation.
        loss = ((estimate - drift).square() * (1 - t[:, None]).square()).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1)
        optimizer.step()
        ema.update(network)
        if step % 100 == 0:
            losses.append({"step": step + 1, "weighted_drift_mse": loss.item()})
    network.load_state_dict(ema.model.state_dict())
    network.eval()
    return losses, optimizer.state_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/dsbm"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", default="gmm")
    parser.add_argument("--initialization", choices=["imf", "ipf"], default="imf")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=6000)
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--simulation-steps", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if (
        min(args.iterations, args.inner_steps, args.particles, args.simulation_steps, args.epsilon)
        <= 0
    ):
        raise ValueError("Counts and diffusion variance must be positive")
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = (
        GaussianMixture(device=device)
        if args.dataset == "gmm"
        else ToyDistribution(args.dataset, device=device)
    )
    source = torch.randn(args.particles, 2, device=device)
    target = (
        data.sample(args.particles)[0]
        if args.initialization == "imf"
        else source + args.epsilon**0.5 * torch.randn_like(source)
    )
    forward, backward = [TimeMLP(args.width, 4).to(device) for _ in range(2)]
    first, records = 0, []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        forward.load_state_dict(state["forward"])
        backward.load_state_dict(state["backward"])
        source, target = state["source"], state["target"]
        first, records = state["iteration"], state["records"]
        torch.set_rng_state(state["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    save_json(
        args.output / "config.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    for iteration in range(first, args.iterations):
        for direction, network in [("backward", backward), ("forward", forward)]:
            history, optimizer = fit_projection(
                network, source, target, args.epsilon, direction, args.inner_steps
            )
            # Exact reset of the prescribed starting marginal prevents endpoint bias
            # from being carried forward on both sides at every outer iteration.
            starting = (
                data.sample(args.particles)[0]
                if direction == "backward"
                else torch.randn(args.particles, 2, device=device)
            )
            ending, trajectory = simulate_projection(
                network, starting, args.epsilon, args.simulation_steps
            )
            source, target = (ending, starting) if direction == "backward" else (starting, ending)
            reference = (
                torch.randn_like(ending) if direction == "backward" else data.sample(len(ending))[0]
            )
            metrics = toy_metrics(ending[:1024], reference[:1024])
            record = {"iteration": iteration + 1, "direction": direction, **metrics}
            records.append(record)
            prefix = args.output / f"iteration_{iteration + 1}_{direction}"
            prefix.mkdir(exist_ok=True)
            scatter_comparison(
                reference, ending, prefix / "samples.png", f"DSBM {direction} {iteration + 1}"
            )
            particle_animation(
                trajectory.cpu(), prefix / "generation.gif", title=f"DSBM {direction}"
            )
            save_json(prefix / "history.json", history)
            save_json(prefix / "metrics.json", record)
            torch.save(
                {"model": network.state_dict(), "optimizer": optimizer}, prefix / "projection.pt"
            )
            print(record, flush=True)
        atomic_checkpoint(
            args.output / "checkpoint.pt",
            {
                "forward": forward.state_dict(),
                "backward": backward.state_dict(),
                "source": source,
                "target": target,
                "iteration": iteration + 1,
                "records": records,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
        )
        save_json(args.output / "metrics.json", records)


if __name__ == "__main__":
    main()
