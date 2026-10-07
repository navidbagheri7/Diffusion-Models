"""Likelihood lessons: analytic GMM checks, learned ODE densities and full DDPM ELBO."""

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import GaussianMixture, ImageDataset, balanced_image_indices
from .likelihood import ddpm_elbo, gaussian_log_density, ode_log_density
from .metrics import image_pixels
from .paths import DiscreteVP, GaussianPath, expand
from .utils import save_json, seed_everything


def analytic_lesson(output, device, samples=256, steps=100):
    generator = torch.Generator(device=device).manual_seed(123)
    data = GaussianMixture(device=device)
    x, _ = data.sample(samples, generator)
    x = x.double()
    path = GaussianPath("vp_linear")
    start, end = 0.0001, 1.0

    def field(x, t):
        a, s = path.coefficients(t)
        f, g = path.sde(x, t)
        return f - 0.5 * expand(g.square(), x) * data.score(x, a, s)

    terminal_a, terminal_s = path.coefficients(x.new_tensor([end]))
    initial_a, initial_s = path.coefficients(x.new_tensor([start]))
    truth = data.log_prob(x, initial_a, initial_s)
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for trace in ["exact", "hutchinson"]:
        density, metadata = ode_log_density(
            field,
            x,
            torch.linspace(start, end, steps + 1, device=device, dtype=x.dtype),
            lambda y: data.log_prob(y, terminal_a, terminal_s),
            trace=trace,
            probe_count=8,
            generator=generator,
        )
        error = density - truth
        record = {
            "trace": trace,
            "probes": metadata["probe_count"],
            "field_calls": metadata["field_calls"],
            "rmse_log_density": float(error.square().mean().sqrt()),
            "mean_absolute_error": float(error.abs().mean()),
            "reference": "Exact GMM density at t=0.0001; exact GMM terminal marginal, not a Gaussian approximation",
        }
        records.append(record)
        axes[0].scatter(truth.cpu(), density.cpu(), s=7, alpha=0.5, label=trace)
        axes[1].hist(error.cpu().numpy(), bins=40, alpha=0.5, label=trace)
    limits = [float(truth.min()), float(truth.max())]
    axes[0].plot(limits, limits, "k--", lw=1)
    axes[0].set(
        xlabel="Analytic log density",
        ylabel="Numerical ODE log density",
        title="Density along the probability-flow ODE",
    )
    axes[1].set(
        xlabel="Numerical minus analytic log density",
        ylabel="Count",
        title="Integration and trace-estimation error",
    )
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "likelihood.png", dpi=170)
    plt.close(fig)
    save_json(
        output / "metrics.json",
        {
            "dataset": "Analytic 8-component GMM",
            "samples": samples,
            "steps": steps,
            "results": records,
            "seed": 123,
        },
    )


def checkpoint_lesson(args):
    from .evaluate import load_run
    from .image_train import load_image_run
    from .targets import convert_prediction
    from .train import prepare_process

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_run = state.get("format") == "image_v2"
    cfg, model, classes, *_extra = (
        load_image_run(args.checkpoint, device) if image_run else load_run(args.checkpoint, device)
    )
    model.eval().requires_grad_(False)
    if cfg.data.kind == "latents":
        raise ValueError(
            "A latent log density alone is not an image likelihood; a probabilistic decoder and variational treatment are required"
        )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if cfg.data.kind == "gmm":
        data = GaussianMixture(cfg.data.components, cfg.data.radius, cfg.data.std, device)
        clean, labels = data.sample(args.samples, generator)
        quantized = False
    else:
        data = ImageDataset(cfg.data, "validation")
        indices = balanced_image_indices(data, args.samples + args.reference_offset)[
            args.reference_offset :
        ]
        if len(indices) != args.samples:
            raise ValueError("Requested likelihood subset exceeds the held-out split")
        clean = torch.stack([data[index][0] for index in indices]).to(device)
        labels = torch.tensor([data[index][1] for index in indices], device=device)
        quantized = True
    labels = labels if classes else None
    discrete = (image_run and cfg.algorithm == "improved_ddpm") or (
        not image_run and cfg.process.kind == "ddpm"
    )
    if discrete:
        config = cfg if image_run else cfg.process
        path = DiscreteVP(
            config.diffusion_steps if image_run else config.steps, config.schedule, device
        )
        bound = ddpm_elbo(
            model,
            path,
            clean,
            config.prediction,
            labels,
            cfg.learned_variance if image_run else False,
            quantized,
            args.timestep_samples,
            generator,
            args.clip_x0
            if args.clip_x0 is not None
            else True
            if image_run
            else cfg.sampler.get("clip_x0", False),
        )
        arrays = {
            key: value.cpu().tolist()
            for key, value in bound.items()
            if isinstance(value, torch.Tensor)
        }
        report = {key: value for key, value in bound.items() if not isinstance(value, torch.Tensor)}
        report.update(
            {
                key + "_mean": float(value.mean())
                for key, value in bound.items()
                if isinstance(value, torch.Tensor)
            }
        )
        save_json(args.output / "per_example.json", arrays)
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = ["terminal_prior_kl_bpd", "decoder_bpd", "transition_kl_bpd"]
        ax.bar(
            ["Terminal prior KL", "Decoder", "Transition KL sum"],
            [report[key + "_mean"] for key in keys],
        )
        ax.set(ylabel="Bits per dimension", title="Complete negative ELBO: all three contributions")
        fig.tight_layout()
        fig.savefig(args.output / "elbo_terms.png", dpi=160)
        plt.close(fig)
    else:
        # Density of the continuous flow at the declared cutoff, with an explicit base.
        if quantized:
            pixels = image_pixels(clean).to(device)
            uniform = torch.rand(clean.shape, device=device, generator=generator)
            clean = (pixels.float() + uniform) / 128 - 1
        if image_run and cfg.algorithm == "edm":
            times = torch.linspace(
                cfg.sigma_min ** (1 / cfg.rho),
                cfg.sigma_max ** (1 / cfg.rho),
                args.steps + 1,
                device=device,
            ).pow(cfg.rho)

            def field(x, t):
                return (x - model(x, t, labels)) / expand(t, x)

            def prior(x):
                return gaussian_log_density(x, cfg.sigma_max)

            scope = "EDM probability-flow density at positive sigma_min, with a Gaussian terminal base; excludes the sampler's final denoising map"
        elif (image_run and cfg.algorithm == "flow") or (
            not image_run and cfg.process.kind == "flow"
        ):
            times = torch.linspace(1, 0, args.steps + 1, device=device)

            def field(x, t):
                return model(x, t, labels)

            prior = gaussian_log_density
            scope = (
                "Continuous neural flow density at generation time 1; numerical change of variables"
            )
        elif not image_run and cfg.process.kind in {
            "vp_linear",
            "vp_cosine",
            "ve",
            "sub_vp",
            "vp_logsnr",
            "vp_exponential",
        }:
            process = prepare_process(cfg.process, device)
            path = process.path
            times = torch.linspace(
                cfg.process.time_eps, 1 - cfg.process.time_eps, args.steps + 1, device=device
            )

            def field(x, t):
                a, s = path.coefficients(t)
                score = convert_prediction(x, model(x, t, labels), a, s, cfg.process.prediction)[2]
                f, g = path.sde(x, t)
                return f - 0.5 * expand(g.square(), x) * score

            terminal_std = (
                float(path.coefficients(times[-1:].to(device))[1])
                if cfg.process.kind == "ve"
                else 1.0
            )

            def prior(x):
                return gaussian_log_density(x, terminal_std)

            scope = "Learned PF-ODE density with declared Gaussian terminal approximation and positive time cutoff"
        else:
            raise ValueError(
                "Likelihood is supported for DDPM, continuous diffusion and flow, not consistency maps"
            )
        trace = (
            args.trace
            if args.trace != "auto"
            else "exact"
            if clean[0].numel() <= 32
            else "hutchinson"
        )
        logp, metadata = ode_log_density(
            field, clean, times, prior, args.solver, trace, args.probes, generator
        )
        report = {
            key: value for key, value in metadata.items() if not isinstance(value, torch.Tensor)
        }
        report.update(
            {
                "mean_log_density_nats": float(logp.mean()),
                "scope": scope,
                "dequantization": "Uniform 8-bit bins, width 1/128 in [-1,1]"
                if quantized
                else None,
            }
        )
        if quantized:
            report["dequantized_density_bpd_estimate"] = float(
                7 - logp.mean() / (clean[0].numel() * math.log(2))
            )
        save_json(args.output / "per_example.json", {"log_density_nats": logp.cpu().tolist()})
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(logp.cpu().numpy(), bins=min(30, len(logp)))
        ax.set(xlabel="Numerical log density (nats)", ylabel="Held-out examples", title=scope)
        fig.tight_layout()
        fig.savefig(args.output / "likelihood.png", dpi=160)
        plt.close(fig)
    checkpoint_hash = getattr(model, "checkpoint_sha256", None)
    if checkpoint_hash is None:
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_hash = digest.hexdigest()
    report.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "training_steps": state.get("step"),
            "samples": len(clean),
            "seed": args.seed,
            "reference_offset": args.reference_offset,
            "dataset": cfg.data.name,
            "conditional": bool(classes),
            "conditioning_scope": "p(x|held-out class label), no CFG; not marginalized over classes"
            if classes
            else "Unconditional model density",
            "is_exact_nll": False,
            "checkpoint_sha256": checkpoint_hash,
        }
    )
    save_json(args.output / "metrics.json", report)
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("assets/reports/likelihood"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "--reference-offset",
        type=int,
        default=0,
        help="Offset in the deterministic balanced held-out image order",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trace", choices=["auto", "exact", "hutchinson"], default="auto")
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--solver", choices=["euler", "heun", "rk4"], default="heun")
    parser.add_argument("--timestep-samples", type=int)
    parser.add_argument(
        "--clip-x0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override reverse-mean clean prediction clipping; default matches sampler",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if min(args.samples, args.steps, args.probes) < 1:
        raise ValueError("Likelihood counts must be positive")
    if args.reference_offset < 0:
        raise ValueError("reference-offset must be nonnegative")
    seed_everything(args.seed, 4)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        checkpoint_lesson(args)
    else:
        analytic_lesson(args.output, args.device, args.samples, args.steps)


if __name__ == "__main__":
    main()
