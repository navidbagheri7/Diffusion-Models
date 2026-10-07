"""Resample a trained checkpoint and compare guidance or samplers."""

import argparse
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .bridges import bridge_training_pair
from .data import GaussianMixture, ImageDataset
from .guidance import AnalyticGMMClassifier
from .metrics import gmm_metrics, image_metrics
from .models import ImageClassifier, TimeMLP, make_model
from .sampling import sample, sample_batched
from .train import prepare_process
from .transport import sample_coupling, sinkhorn_plan
from .utils import save_json, seed_everything
from .visualize import image_artifacts, particle_animation, scatter_comparison


def load_run(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    if cfg.conditional:
        if cfg.data.kind == "gmm":
            classes = cfg.data.components
        else:
            classes = len(ImageDataset(cfg.data).class_names)
    else:
        classes = 0
    model = make_model(cfg.model, classes).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["ema"])
    return cfg, model, classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--reference-seed",
        type=int,
        help="Independent reference RNG; prevents sampler RNG consumption changing the GMM reference",
    )
    parser.add_argument(
        "--method", choices=["ddpm", "ddim", "ode", "sde", "bridge_sde", "predictor_corrector"]
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--solver", choices=["euler", "heun", "rk4", "rk45", "euler_maruyama", "stochastic_heun"]
    )
    parser.add_argument("--eta", type=float)
    parser.add_argument("--class-label", type=int)
    parser.add_argument("--guidance", choices=["cfg", "classifier"], default="cfg")
    parser.add_argument("--classifier-kind", choices=["learned", "analytic"], default="learned")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--skip-image-metrics", action="store_true")
    args = parser.parse_args()
    if args.scale < 0:
        raise ValueError("Guidance examples use nonnegative scales")
    seed_everything(args.seed)
    device = torch.device(args.device)
    cfg, model, classes = load_run(args.checkpoint, device)
    process = prepare_process(cfg.process, device)
    if args.class_label is not None and not 0 <= args.class_label < classes:
        raise ValueError("Class label is outside the trained model's label range")
    sampler = cfg.sampler
    if args.method:
        sampler.method = args.method
    if args.steps:
        sampler.steps = args.steps
    if args.solver:
        sampler[
            "sde_solver" if args.solver in {"euler_maruyama", "stochastic_heun"} else "ode_solver"
        ] = args.solver
    if args.eta is not None:
        sampler.eta = args.eta
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (
        (args.samples, 2)
        if cfg.data.kind == "gmm"
        else (args.samples, cfg.data.get("channels", 3), cfg.data.image_size, cfg.data.image_size)
    )
    initial = torch.randn(shape, device=device, generator=generator) * (
        process.path.prior_std() if hasattr(process, "path") else 1
    )
    labels = (
        torch.full((args.samples,), args.class_label, device=device, dtype=torch.long)
        if args.class_label is not None
        else None
    )
    mixture = (
        GaussianMixture(cfg.data.components, cfg.data.radius, cfg.data.std, device)
        if cfg.data.kind == "gmm"
        else None
    )
    if process.kind == "bridge":
        endpoint_generator = torch.Generator(device=device).manual_seed(cfg.seed + 700)
        source = torch.randn(
            process.bridge_particles, 2, device=device, generator=endpoint_generator
        )
        target = mixture.sample(process.bridge_particles, endpoint_generator)[0]
        plan = sinkhorn_plan(source, target, process.bridge_epsilon)
        z, y = sample_coupling(source, target, plan, args.samples, generator)
        initial = bridge_training_pair(
            z,
            y,
            torch.full((args.samples,), process.time_eps, device=device),
            process.bridge_epsilon,
        )[0]
    classifier = None
    if args.guidance == "classifier":
        if labels is None:
            raise ValueError("Classifier guidance requires --class-label")
        if args.classifier_kind == "analytic":
            if mixture is None or not hasattr(process, "path"):
                raise ValueError("Analytic classifiers require a GMM diffusion process")
            classifier = AnalyticGMMClassifier(mixture, process.path)
        else:
            classifier = (
                TimeMLP(cfg.model.width, cfg.model.depth, output_dim=classes).to(device)
                if mixture
                else ImageClassifier(
                    cfg.model.get("width", 64), classes, cfg.data.get("channels", 3)
                ).to(device)
            )
            weights = Path(args.checkpoint).parent / "classifier.pt"
            classifier.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
            classifier.eval().requires_grad_(False)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sampling_function = sample if mixture else sample_batched
    network_calls = [0]

    def count_call(_module, _inputs, _output):
        network_calls[0] += 1

    hook = model.register_forward_hook(count_call)
    samples, trajectory = sampling_function(
        model,
        process,
        sampler,
        initial,
        process.prediction,
        labels,
        args.scale if classifier is None else 0.0,
        classifier,
        args.scale,
        generator,
    )
    hook.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    conditional_reference = labels is not None and args.scale > 0
    if mixture:
        # Conditional references are the selected component, not the full mixture.
        reference = mixture.sample(
            args.samples,
            torch.Generator(device=device).manual_seed(args.reference_seed)
            if args.reference_seed is not None
            else generator,
            labels if conditional_reference else None,
        )[0]
        metric_mixture = mixture
        if conditional_reference:
            metric_mixture = GaussianMixture(1, 0, cfg.data.std, device)
            metric_mixture.means = mixture.means[args.class_label : args.class_label + 1]
        metrics = gmm_metrics(samples, reference, metric_mixture)
        scatter_comparison(
            reference, samples, directory / "samples.png", f"{args.guidance}, scale={args.scale}"
        )
        particle_animation(
            trajectory, directory / "generation.gif", title=f"{args.guidance}, scale={args.scale}"
        )
    else:
        data = ImageDataset(cfg.data, "validation")
        selected = [
            i
            for i, (_, label) in enumerate(data.files)
            if not conditional_reference or label == args.class_label
        ]
        if not selected:
            raise ValueError("No validation images of the requested class selected")
        reference = torch.stack([data[i][0] for i in selected[: args.samples]])
        image_artifacts(samples, trajectory, directory)
        if args.skip_image_metrics:
            metrics = {"image_metrics": "disabled by argument"}
        elif len(reference) < 2:
            metrics = {
                "image_metrics": "requires at least two selected reference images",
                "real_count": len(reference),
                "generated_count": len(samples),
            }
        else:
            metrics = image_metrics(reference, samples, device)
    metrics.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "seed": args.seed,
            "reference_seed": args.reference_seed,
            "network_calls_during_sampling": network_calls[0],
            "nfe_per_image": network_calls[0] if mixture else None,
            "sample_count": args.samples,
            "sampling_seconds": seconds,
            "guidance": args.guidance,
            "guidance_scale": args.scale,
            "class_label": args.class_label,
            "classifier_kind": args.classifier_kind if classifier is not None else None,
            "sampler": OmegaConf.to_container(sampler),
            "process": process.kind,
            "prediction": process.prediction,
        }
    )
    save_json(directory / "metrics.json", metrics)
    torch.save(
        {"samples": samples.cpu(), "reference": reference.cpu(), "trajectory": trajectory[:, :64]},
        directory / "samples.pt",
    )
    print(metrics)


if __name__ == "__main__":
    main()
