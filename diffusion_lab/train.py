"""Hydra training entrypoint. Supports independent runs or torchrun DDP."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .bridges import bridge_training_pair
from .data import GaussianMixture, ImageDataset
from .guidance import drop_labels
from .metrics import gmm_metrics, image_metrics
from .models import ImageClassifier, TimeMLP, make_model
from .paths import expand, flow_coefficients, make_path
from .sampling import sample, sample_batched
from .targets import training_target
from .transport import conditional_pairing, exact_pairing, sample_coupling, sinkhorn_plan
from .utils import EMA, atomic_checkpoint, save_json, seed_everything
from .visualize import image_artifacts, loss_plot, particle_animation, scatter_comparison


def prepare_process(config, device):
    process = SimpleNamespace(**OmegaConf.to_container(config, resolve=True))
    if process.kind not in {"flow", "bridge"}:
        process.path = make_path(config, device)
    return process


def perturb(x0, labels, process, generator=None, bridge_endpoints=None, conditional=False):
    def noise_like(x):
        return torch.randn(x.shape, device=x.device, generator=generator)

    noise = noise_like(x0)
    if process.kind == "ddpm":
        indices = torch.randint(process.steps, (len(x0),), device=x0.device, generator=generator)
        t = (indices.float() + 1) / process.steps
    else:
        t = torch.rand(len(x0), device=x0.device, generator=generator)
        if process.kind != "flow":
            t = process.time_eps + (1 - 2 * process.time_eps) * t
    if process.kind == "flow":
        if process.coupling == "reflow":
            source, target, _ = bridge_endpoints
            index = torch.randint(len(source), (len(x0),), device=x0.device, generator=generator)
            noise, x0 = source[index], target[index]
        elif process.coupling == "ot":
            if len(x0) > 256:
                raise ValueError("Educational minibatch OT is capped at 256 examples")
            # Pair within the original classes before classifier-free label dropout.
            # Global OT remains appropriate for unconditional training.
            noise = (
                conditional_pairing(x0.flatten(1), noise.flatten(1), labels)
                if conditional
                else exact_pairing(x0.flatten(1), noise.flatten(1))
            ).reshape_as(x0)
        elif process.coupling != "independent":
            raise ValueError("Flow coupling must be independent or ot")
        a, s, da, ds = flow_coefficients(t, process.flow_path)
        return (
            expand(a, x0) * x0 + expand(s, x0) * noise,
            t,
            expand(da, x0) * x0 + expand(ds, x0) * noise,
            1.0,
        )
    if process.kind == "bridge":
        source, target, plan = bridge_endpoints
        z, y = sample_coupling(source, target, plan, len(x0), generator)
        x, drift, velocity = bridge_training_pair(z, y, t, process.bridge_epsilon, noise)
        return x, t, drift if process.prediction == "bridge_drift" else velocity, 1.0
    alpha, sigma = process.path.coefficients(t)
    x = expand(alpha, x0) * x0 + expand(sigma, x0) * noise
    target, weight = training_target(x0, noise, alpha, sigma, process.prediction)
    return x, t, target, weight


def validate_configuration(cfg):
    if (
        min(
            cfg.steps,
            cfg.batch_size,
            cfg.log_every,
            cfg.validate_every,
            cfg.validation_samples,
            cfg.evaluation_samples,
            cfg.cpu_threads,
            cfg.grad_clip,
            cfg.lr,
        )
        <= 0
    ):
        raise ValueError("Training counts, learning rate, and clipping threshold must be positive")
    if cfg.workers < 0 or cfg.classifier_steps < 0 or not 0 <= cfg.ema_decay < 1:
        raise ValueError("Invalid worker/classifier count or EMA decay")
    if cfg.sampler.steps < 1:
        raise ValueError("Sampling needs at least one integration step")
    if cfg.evaluation_samples < 2:
        raise ValueError("Distribution metrics require at least two evaluation samples")
    if cfg.model.get("width", 64) < 1 or (
        cfg.model.get("kind") == "mlp" and cfg.model.get("width", 64) % 2
    ):
        raise ValueError("Model width must be positive, and MLP embedding width must be even")
    if cfg.data.kind == "images" and (cfg.data.image_size < 4 or cfg.data.image_size % 4):
        raise ValueError("Image sizes must be positive multiples of four")
    if "time_eps" in cfg.process and not 0 < cfg.process.time_eps < 0.5:
        raise ValueError("time_eps must lie in (0,0.5)")
    if cfg.process.kind == "flow" and cfg.process.prediction != "velocity":
        raise ValueError("Flow matching predicts generation-time velocity")
    if cfg.process.kind == "flow":
        if cfg.sampler.method != "ode":
            raise ValueError("Flow matching uses an ODE sampler")
        if cfg.process.coupling not in {"independent", "ot", "reflow"}:
            raise ValueError("Unknown endpoint coupling")
        if cfg.process.coupling == "ot" and cfg.batch_size > 256:
            raise ValueError("Exact minibatch OT is limited to 256 examples")
        if cfg.process.coupling == "reflow" and (
            cfg.data.kind != "gmm" or cfg.conditional or not cfg.reflow_checkpoint
        ):
            raise ValueError("Reflow requires an unconditional GMM teacher checkpoint")
    if cfg.process.kind == "bridge":
        if cfg.data.kind != "gmm" or cfg.conditional:
            raise ValueError("Empirical bridge training supports unconditional GMM experiments")
        if cfg.process.prediction not in {"velocity", "bridge_drift"}:
            raise ValueError("Bridge prediction must be velocity or bridge_drift")
        expected = "bridge_sde" if cfg.process.prediction == "bridge_drift" else "ode"
        if cfg.sampler.method != expected:
            raise ValueError(f"Bridge target {cfg.process.prediction} requires sampler {expected}")
    if cfg.process.kind == "ddpm":
        if (
            cfg.sampler.method not in {"ddpm", "ddim", "dpm_1", "dpm_2m", "dpmpp_1", "dpmpp_2m"}
            or cfg.sampler.steps > cfg.process.steps
        ):
            raise ValueError("Discrete VP requires DDPM/DDIM with steps no greater than its table")
    elif cfg.process.kind not in {"flow", "bridge"}:
        if cfg.sampler.method not in {
            "ode",
            "sde",
            "predictor_corrector",
            "dpm_1",
            "dpm_2m",
            "dpmpp_1",
            "dpmpp_2m",
        }:
            raise ValueError("Continuous diffusion requires an ODE/SDE/predictor-corrector sampler")
    if cfg.process.kind not in {"flow", "bridge"} and cfg.process.prediction not in {
        "x0",
        "epsilon",
        "score",
        "v",
    }:
        raise ValueError("Unknown diffusion prediction target")
    if cfg.data.kind == "gmm" and cfg.model.get("kind") != "mlp" and "_target_" not in cfg.model:
        raise ValueError("GMM experiments require the MLP")
    if (
        cfg.data.kind == "images"
        and cfg.model.get("kind") not in {"unet", "image_unet"}
        and "_target_" not in cfg.model
    ):
        raise ValueError("Image experiments require the U-Net")
    if cfg.conditional and not 0 < cfg.label_dropout < 1:
        raise ValueError("Conditional/CFG training requires a nonzero null-token dropout below one")
    if cfg.classifier_steps and (not cfg.conditional or cfg.process.kind in {"flow", "bridge"}):
        raise ValueError("Noisy classifier training requires conditional diffusion")


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
if not CONFIG_ROOT.is_dir():
    CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


@hydra.main(version_base="1.3", config_path=str(CONFIG_ROOT), config_name="train")
def main(cfg: DictConfig):
    validate_configuration(cfg)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(
        f"cuda:{local_rank}" if world > 1 and cfg.device.startswith("cuda") else cfg.device
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed_everything(cfg.seed + rank, cfg.cpu_threads)
    directory = Path(to_absolute_path(cfg.run_dir))
    directory.mkdir(parents=True, exist_ok=True)
    process = prepare_process(cfg.process, device)
    mixture, endpoints, loader, validation_loader = None, None, None, None
    if cfg.data.kind == "gmm":
        mixture = GaussianMixture(cfg.data.components, cfg.data.radius, cfg.data.std, device)
        classes = cfg.data.components if cfg.conditional else 0
        if process.kind == "bridge":
            generator = torch.Generator(device=device).manual_seed(cfg.seed + 700)
            source = torch.randn(process.bridge_particles, 2, device=device, generator=generator)
            target = mixture.sample(process.bridge_particles, generator)[0]
            plan = sinkhorn_plan(source, target, process.bridge_epsilon)
            endpoints = source, target, plan
        if process.kind == "flow" and process.coupling == "reflow":
            if cfg.conditional or not cfg.reflow_checkpoint:
                raise ValueError("Reflow requires an unconditional teacher checkpoint")
            teacher_state = torch.load(
                to_absolute_path(cfg.reflow_checkpoint), map_location=device, weights_only=False
            )
            teacher_cfg = OmegaConf.create(teacher_state["config"])
            if (
                teacher_cfg.process.kind != "flow"
                or teacher_cfg.conditional
                or teacher_cfg.data.kind != "gmm"
            ):
                raise ValueError("This reflow example requires an unconditional GMM flow teacher")
            teacher = make_model(teacher_cfg.model).to(device).eval().requires_grad_(False)
            teacher.load_state_dict(teacher_state["ema"])
            teacher_process = prepare_process(teacher_cfg.process, device)
            source = torch.randn(process.reflow_particles, 2, device=device)
            targets = [
                sample(teacher, teacher_process, teacher_cfg.sampler, z, "velocity")[0]
                for z in source.split(256)
            ]
            endpoints = source, torch.cat(targets), None
    else:
        train_data, validation_data = ImageDataset(cfg.data), ImageDataset(cfg.data, "validation")
        if train_data.class_names != validation_data.class_names:
            raise ValueError("Train/validation class folders do not match")
        classes = len(train_data.class_names) if cfg.conditional else 0
        if cfg.conditional and classes < 2:
            raise ValueError("Conditional image training needs at least two labeled class folders")
        distributed_sampler = (
            DistributedSampler(train_data, shuffle=True, seed=cfg.seed) if world > 1 else None
        )
        loader = DataLoader(
            train_data,
            batch_size=cfg.batch_size,
            shuffle=distributed_sampler is None,
            sampler=distributed_sampler,
            num_workers=cfg.workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(cfg.seed + rank),
            drop_last=True,
        )
        if not len(loader):
            raise ValueError("Training set is smaller than the batch size")
        validation_loader = DataLoader(
            validation_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers
        )
    # Identical initialization across DDP ranks, then independent training randomness.
    torch.manual_seed(cfg.seed)
    network = make_model(cfg.model, classes).to(device)
    ema = EMA(network, cfg.ema_decay)
    model = (
        DistributedDataParallel(network, device_ids=[local_rank] if device.type == "cuda" else None)
        if world > 1
        else network
    )
    torch.manual_seed(cfg.seed + rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history, first_step = [], 0
    if cfg.resume:
        checkpoint = torch.load(
            to_absolute_path(cfg.resume), map_location=device, weights_only=False
        )
        network.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_step = checkpoint["step"]
        history = checkpoint["history"]
    if rank == 0:
        OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
        save_json(
            directory / "environment.json",
            {
                "torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "world_size": world,
                "parameters": sum(p.numel() for p in network.parameters()),
                "train_count": len(train_data) if loader else "online GMM",
                "validation_count": len(validation_data) if loader else cfg.validation_samples,
            },
        )
    iterator, epoch = iter(loader) if loader else None, 0

    def batch(validation=False, generator=None):
        nonlocal iterator, epoch
        if mixture is not None:
            return mixture.sample(cfg.batch_size, generator)
        if validation:
            raise RuntimeError("Use the held-out loader for image validation")
        try:
            x, y = next(iterator)
        except StopIteration:
            epoch += 1
            if world > 1:
                loader.sampler.set_epoch(epoch)
            iterator = iter(loader)
            x, y = next(iterator)
        return x.to(device), y.to(device)

    @torch.no_grad()
    def validation():
        ema.model.eval()
        generator = torch.Generator(device=device).manual_seed(cfg.seed + 10000)
        total, count = 0.0, 0
        batches = (
            (
                (mixture.sample(cfg.batch_size, generator))
                for _ in range(max(1, cfg.validation_samples // cfg.batch_size))
            )
            if mixture
            else iter(validation_loader)
        )
        for x0, labels in batches:
            x0, labels = x0.to(device), labels.to(device)
            x, t, target, weight = perturb(
                x0, labels, process, generator, endpoints, conditional=bool(classes)
            )
            estimate = ema.model(x, t, labels if classes else None)
            total += ((estimate - target).square() * weight).flatten(1).mean(1).sum().item()
            count += len(x0)
            if count >= cfg.validation_samples:
                break
        return total / count

    started, accumulated, logged = time.perf_counter(), 0.0, 0
    for step in range(first_step + 1, cfg.steps + 1):
        model.train()
        x0, labels = batch()
        x, t, target, weight = perturb(
            x0, labels, process, bridge_endpoints=endpoints, conditional=bool(classes)
        )
        condition = drop_labels(labels, classes, cfg.label_dropout) if classes else None
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"
        ):
            estimate = model(x, t, condition)
            loss = ((estimate.float() - target).square() * weight).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        ema.update(network)
        accumulated += loss.item()
        logged += 1
        if step % cfg.log_every == 0 or step == cfg.steps:
            row = {
                "step": step,
                "train_loss": accumulated / logged,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if world > 1:
                value = torch.tensor(row["train_loss"], device=device)
                dist.all_reduce(value)
                row["train_loss"] = value.item() / world
            if rank == 0:
                history.append(row)
                print(json.dumps(row), flush=True)
            accumulated, logged = 0.0, 0
        if step % cfg.validate_every == 0 or step == cfg.steps:
            value = validation()
            if rank == 0:
                history.append({"step": step, "validation_loss": value})
                print(f"validation step={step}: {value:.6f}", flush=True)
                atomic_checkpoint(
                    directory / "checkpoint.pt",
                    {
                        "model": network.state_dict(),
                        "ema": ema.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "history": history,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                )
    if world > 1:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
    if rank == 0:
        classifier = None
        if cfg.classifier_steps:
            classifier = (
                TimeMLP(
                    cfg.model.get("width", 64), cfg.model.get("depth", 3), output_dim=classes
                ).to(device)
                if mixture
                else ImageClassifier(
                    cfg.model.get("width", 64), classes, cfg.data.get("channels", 3)
                ).to(device)
            )
            opt = torch.optim.AdamW(classifier.parameters(), lr=cfg.lr)
            for _ in range(cfg.classifier_steps):
                x0, labels = batch()
                x, t, _, _ = perturb(x0, labels, process)
                loss = F.cross_entropy(classifier(x, t), labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            classifier.eval().requires_grad_(False)
            atomic_checkpoint(directory / "classifier.pt", classifier.state_dict())
        loss_plot(history, directory / "loss.png")
        save_json(directory / "history.json", history)
        evaluate_run(
            cfg, ema.model, process, mixture, validation_loader, directory, device, endpoints
        )
    if world > 1:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
        dist.destroy_process_group()


@torch.no_grad()
def evaluate_run(
    cfg, model, process, mixture, validation_loader, directory, device, endpoints=None
):
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 20000)
    n = cfg.evaluation_samples
    shape = (
        (n, 2)
        if mixture
        else (n, cfg.data.get("channels", 3), cfg.data.image_size, cfg.data.image_size)
    )
    std = process.path.prior_std() if hasattr(process, "path") else 1.0
    initial = torch.randn(shape, device=device, generator=generator) * std
    if process.kind == "bridge":
        # Correct start marginal at epsilon time, rather than pretending t=0 is nonsingular.
        z, y = sample_coupling(*endpoints, n, generator)
        t = torch.full((n,), process.time_eps, device=device)
        initial = bridge_training_pair(
            z, y, t, process.bridge_epsilon, torch.randn(shape, device=device, generator=generator)
        )[0]
    started = time.perf_counter()
    sampling_function = sample if mixture else sample_batched
    samples, trajectory = sampling_function(
        model, process, cfg.sampler, initial, process.prediction, generator=generator
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sampling_seconds = time.perf_counter() - started
    if mixture:
        reference = mixture.sample(n, generator)[0]
        metrics = gmm_metrics(samples, reference, mixture)
        scatter_comparison(reference, samples, directory / "samples.png")
        particle_animation(
            trajectory, directory / "generation.gif", title=f"{process.kind}/{cfg.sampler.method}"
        )
    else:
        real = []
        count = 0
        for x, _ in validation_loader:
            real.append(x)
            count += len(x)
            if count >= n:
                break
        reference = torch.cat(real)[:n]
        image_artifacts(samples, trajectory, directory)
        metrics = (
            image_metrics(reference, samples, device, cfg.metric_batch_size)
            if cfg.image_metrics
            else {"image_metrics": "disabled by configuration"}
        )
    metrics.update(
        {
            "seed": cfg.seed,
            "training_steps": cfg.steps,
            "sample_count": n,
            "sampling_seconds": sampling_seconds,
            "sampler": OmegaConf.to_container(cfg.sampler),
            "prediction": process.prediction,
            "process": process.kind,
            "conditional_evaluation": False,
        }
    )
    save_json(directory / "metrics.json", metrics)
    torch.save(
        {"samples": samples.cpu(), "reference": reference.cpu(), "trajectory": trajectory[:, :64]},
        directory / "samples.pt",
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
