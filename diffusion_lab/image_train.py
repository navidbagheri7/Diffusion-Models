"""Long-run image training for EDM, consistency, flow and Improved DDPM.

Shares the data/model/configuration API with the small mathematical lessons.
Checkpoints include optimizer, EMA, sampler and RNG states. torchrun is supported.
"""

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image

from .consistency import (
    ConsistencyModel,
    consistency_loss,
    consistency_sample,
    improved_consistency_loss,
    improved_scales,
)
from .data import balanced_image_indices
from .dpm_solver import dpm_sample
from .edm import EDMPreconditioner, edm_loss, edm_sample
from .guidance import drop_labels
from .improved_ddpm import LossSecondMomentSampler, improved_loss
from .latents import (
    FrozenAutoencoder,
    decode_trajectory,
    image_or_latent_dataset,
    posterior_samples,
)
from .models import make_model
from .paths import DiscreteVP, expand, flow_coefficients
from .progressive import nested_teacher_sample, progressive_loss, progressive_sample
from .sampling import discrete_sample
from .solvers import integrate
from .train import CONFIG_ROOT
from .utils import EMA, atomic_checkpoint, save_json, seed_everything
from .visualize import image_artifacts, loss_plot


def build_image_model(cfg, classes):
    network = make_model(cfg.model, classes)
    if cfg.algorithm == "edm":
        return EDMPreconditioner(network, cfg.sigma_data)
    if cfg.algorithm in {"consistency", "distillation", "improved_consistency"}:
        return ConsistencyModel(network, cfg.sigma_min, cfg.sigma_data)
    return network


def load_image_run(path, device):
    # Hash and load the same inode, even if a trainer replaces its latest file.
    with Path(path).open("rb") as handle:
        state = torch.load(handle, map_location=device, weights_only=False)
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if state.get("format") != "image_v2":
        raise ValueError("Expected an image_train checkpoint")
    cfg = OmegaConf.create(state["config"])
    model = build_image_model(cfg, state["classes"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(state["ema"])
    model.checkpoint_sha256 = digest.hexdigest()
    model.latent_manifest = state.get("latent_manifest")
    if cfg.data.kind == "latents" and model.latent_manifest is None:
        raise ValueError("Latent checkpoint must embed its representation manifest")
    return cfg, model, state["classes"], state["step"]


@torch.no_grad()
def sample_images(
    cfg,
    model,
    noise,
    labels=None,
    scale=1.0,
    method=None,
    steps=None,
    generator=None,
    classifier=None,
    classifier_scale=0,
    consistency_sigmas=None,
    clip_consistency=False,
):
    steps = steps or cfg.sampling_steps
    if cfg.algorithm == "edm":
        return edm_sample(
            model,
            noise,
            steps,
            cfg.sigma_min,
            cfg.sigma_max,
            cfg.rho,
            method or "heun",
            labels,
            scale,
            generator=generator,
            classifier=classifier,
            classifier_scale=classifier_scale,
        )
    if cfg.algorithm in {"consistency", "distillation", "improved_consistency"}:
        return consistency_sample(
            model,
            noise,
            steps,
            cfg.sigma_max,
            labels,
            scale,
            generator,
            sigmas=consistency_sigmas,
            clip_denoised=clip_consistency,
        )
    if cfg.algorithm == "progressive":
        if steps != cfg.student_steps:
            raise ValueError("A progressively distilled stage uses its trained step grid")
        return progressive_sample(
            model,
            DiscreteVP(cfg.diffusion_steps, cfg.schedule, noise.device),
            noise,
            steps,
            labels,
            scale,
        )
    if cfg.algorithm == "flow":
        from .guidance import classifier_free

        calls = 0

        def field(x, t):
            nonlocal calls
            calls += 2 if labels is not None and scale not in {0, 1} else 1
            return classifier_free(model, x, t, labels, scale)

        result, history = integrate(
            field, noise, torch.linspace(0, 1, steps + 1, device=noise.device), method or "heun"
        )
        return result, history[:, :32], calls
    path = DiscreteVP(cfg.diffusion_steps, cfg.schedule, noise.device)
    if method == "nested_ddim":
        return nested_teacher_sample(model, cfg, path, noise, steps, labels, scale)
    method = method or "dpmpp_2m"
    if method.startswith("dpm") and method != "ddpm":
        return dpm_sample(model, path, noise, cfg.prediction, steps, method, labels, scale, True)
    result, history = discrete_sample(
        model,
        path,
        noise,
        cfg.prediction,
        method,
        steps,
        labels,
        scale,
        clip_x0=True,
        generator=generator,
        learned_variance=cfg.learned_variance,
    )
    calls = steps * (2 if labels is not None and scale not in {0, 1} else 1)
    return result, history[:, :32], calls


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


@hydra.main(version_base="1.3", config_path=str(CONFIG_ROOT), config_name="image_train")
def main(cfg):
    if cfg.algorithm not in {
        "edm",
        "consistency",
        "distillation",
        "improved_consistency",
        "progressive",
        "flow",
        "improved_ddpm",
    }:
        raise ValueError("Unknown image algorithm")
    if (
        min(
            cfg.steps,
            cfg.batch_size,
            cfg.accumulation,
            cfg.save_every,
            cfg.preview_every,
            cfg.log_every,
            cfg.lr,
            cfg.sampling_steps,
        )
        <= 0
    ):
        raise ValueError("Training counts and learning rate must be positive")
    if cfg.learned_variance and cfg.algorithm != "improved_ddpm":
        raise ValueError("Learned variance is specific to discrete DDPM")
    if cfg.model.get("out_channels") is not None and cfg.model.out_channels != cfg.data.channels * (
        2 if cfg.learned_variance else 1
    ):
        raise ValueError("Output channels must equal data channels, doubled for learned variance")
    if cfg.model.get("channels") is not None and cfg.model.channels != cfg.data.channels:
        raise ValueError("Model input channels must equal data channels")
    if cfg.algorithm == "improved_ddpm" and cfg.objective == "simple" and cfg.learned_variance:
        raise ValueError("A variance head requires hybrid or VLB supervision")
    if cfg.timestep_sampling not in {"uniform", "loss_second_moment"}:
        raise ValueError("Unknown timestep sampling scheme")
    if not 0 < cfg.sigma_min < cfg.sigma_max or cfg.rho <= 0 or cfg.sigma_data <= 0:
        raise ValueError("Invalid noise/preconditioning bounds")
    if cfg.conditional and not 0 < cfg.label_dropout < 1:
        raise ValueError("CFG requires label dropout in (0,1)")
    if cfg.algorithm == "distillation" and not cfg.teacher:
        raise ValueError("Consistency distillation requires an EDM teacher")
    if cfg.algorithm == "progressive" and (
        not cfg.teacher or cfg.prediction != "x0" or cfg.sampling_steps != cfg.student_steps
    ):
        raise ValueError(
            "Progressive distillation needs a teacher, x0 prediction and matching sampling/student_steps"
        )
    if cfg.algorithm == "improved_consistency" and cfg.teacher:
        raise ValueError("iCT uses an online target, not a diffusion teacher")
    target_decay = cfg.get("target_ema_decay")
    if target_decay is not None and (
        cfg.algorithm not in {"consistency", "distillation"} or not 0 <= target_decay < 1
    ):
        raise ValueError("A separate target EMA is only supported for CT/CD, with decay in [0,1)")
    if (
        cfg.algorithm in {"consistency", "distillation", "improved_consistency"}
        and cfg.model.get("dropout", 0) != 0
    ):
        raise ValueError("These consistency presets require dropout=0 for shared target semantics")
    world, rank, local_rank = (
        int(os.environ.get(k, v))
        for k, v in [("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")]
    )
    device = torch.device(f"cuda:{local_rank}" if world > 1 else cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed_everything(cfg.seed + rank, cfg.cpu_threads)
    directory = Path(to_absolute_path(cfg.run_dir))
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoint.pt").exists() and not cfg.resume:
        raise ValueError("Run directory already has weights; supply resume or choose a new run_dir")
    if cfg.lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be constant or cosine")
    if not 0 <= cfg.ema_decay < 1 or cfg.grad_clip <= 0:
        raise ValueError("Invalid EMA decay or gradient clipping threshold")
    train_data_config = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    # Augment on the device using checkpointed RNG, avoiding worker RNG ambiguity.
    train_data_config.horizontal_flip = False
    is_latent = cfg.data.kind == "latents"
    if is_latent and cfg.data.get("horizontal_flip", False):
        raise ValueError(
            "Latent augmentation uses cached pixel-flip encodings, not latent-space flips"
        )
    if is_latent and cfg.algorithm == "improved_ddpm":
        raise ValueError(
            "Discrete image decoder/variance likelihood is not a quantized latent likelihood; use EDM or flow for latent presets"
        )
    data, validation = (
        image_or_latent_dataset(train_data_config),
        image_or_latent_dataset(cfg.data, "validation"),
    )
    latent_manifest = data.manifest if is_latent else None
    codec = (
        FrozenAutoencoder(
            latent_manifest["autoencoder"]["path"], device, latent_manifest["autoencoder"]
        )
        if is_latent and rank == 0
        else None
    )
    if data.class_names != validation.class_names:
        raise ValueError("Train and validation labels differ")
    classes = len(data.class_names) if cfg.conditional else 0
    if cfg.conditional and classes < 2:
        raise ValueError("Conditional training requires labeled classes")
    loader_generator = torch.Generator().manual_seed(cfg.seed + rank)
    sampler = DistributedSampler(data, seed=cfg.seed) if world > 1 else None
    loader = DataLoader(
        data,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        persistent_workers=cfg.workers > 0,
    )
    if not len(loader):
        raise ValueError("Batch size exceeds dataset size")
    torch.manual_seed(cfg.seed)
    network = build_image_model(cfg, classes).to(device)
    ema = EMA(network, cfg.ema_decay)
    train_model = (
        DistributedDataParallel(network, device_ids=[local_rank] if device.type == "cuda" else None)
        if world > 1
        else network
    )
    optimizer = torch.optim.AdamW(
        train_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
    )
    path = (
        DiscreteVP(cfg.diffusion_steps, cfg.schedule, device)
        if cfg.algorithm in {"improved_ddpm", "progressive"}
        else None
    )
    time_sampler = (
        LossSecondMomentSampler(cfg.diffusion_steps, device=device)
        if path
        and cfg.algorithm == "improved_ddpm"
        and cfg.timestep_sampling == "loss_second_moment"
        else None
    )
    teacher = None
    if cfg.teacher:
        teacher_cfg, teacher, teacher_classes, _ = load_image_run(
            to_absolute_path(cfg.teacher), device
        )
        if (
            teacher_cfg.algorithm
            not in ({"improved_ddpm", "progressive"} if cfg.algorithm == "progressive" else {"edm"})
            or teacher_classes != classes
            or teacher_cfg.data.channels != cfg.data.channels
        ):
            raise ValueError("Teacher must have a compatible algorithm and class/channel count")
        if cfg.algorithm == "progressive":
            if (
                teacher_cfg.diffusion_steps != cfg.diffusion_steps
                or teacher_cfg.schedule != cfg.schedule
            ):
                raise ValueError("Progressive teacher and student need the same diffusion schedule")
            if (
                teacher_cfg.algorithm == "progressive"
                and teacher_cfg.student_steps != 2 * cfg.student_steps
            ):
                raise ValueError("Each progressive stage halves its teacher's step count")
            initial = {
                key: value
                for key, value in teacher.state_dict().items()
                if key in network.state_dict()
                and value.shape == network.state_dict()[key].shape
                and (teacher_cfg.prediction == "x0" or not key.startswith("output."))
            }
            network.load_state_dict(initial, strict=False)
            ema.model.load_state_dict(network.state_dict())
        if cfg.algorithm == "distillation":
            network.network.load_state_dict(teacher.network.state_dict())
            ema.model.load_state_dict(network.state_dict())
    target_ema = EMA(network, target_decay) if target_decay is not None else ema
    first_step, history, epoch, offset = 0, [], 0, 0
    resume_rng = None
    if cfg.resume:
        state = torch.load(to_absolute_path(cfg.resume), map_location=device, weights_only=False)
        if state["classes"] != classes or state["config"]["algorithm"] != cfg.algorithm:
            raise ValueError("Resume checkpoint algorithm/classes mismatch")
        current_config = OmegaConf.to_container(cfg, resolve=True)
        restored_fields = [
            "data",
            "model",
            "batch_size",
            "accumulation",
            "workers",
            "seed",
            "conditional",
            "label_dropout",
            "sigma_data",
            "sigma_min",
            "sigma_max",
            "p_mean",
            "p_std",
            "prediction",
            "schedule",
            "diffusion_steps",
            "flow_path",
            "learned_variance",
            "objective",
            "hybrid_weight",
            "timestep_sampling",
            "lr",
            "lr_schedule",
            "warmup_steps",
            "weight_decay",
            "grad_clip",
            "ema_decay",
            "consistency_scales_min",
            "consistency_scales_max",
            "teacher",
        ]
        if cfg.algorithm == "progressive":
            restored_fields.append("student_steps")
        changed = [key for key in restored_fields if current_config[key] != state["config"][key]]
        defaults = {
            "consistency_weighting": "uniform",
            "consistency_loss_norm": "mse",
            "target_ema_decay": None,
        }
        changed.extend(
            key
            for key, default in defaults.items()
            if current_config.get(key, default) != state["config"].get(key, default)
        )
        if changed:
            raise ValueError(f"Resume changes the saved training setup: {', '.join(changed)}")
        if is_latent and state.get("latent_manifest") != latent_manifest:
            raise ValueError("Resume changes the cached latent representation")
        network.load_state_dict(state["model"])
        ema.model.load_state_dict(state["ema"])
        if target_ema is not ema:
            target_ema.model.load_state_dict(state["target_ema"])
        if teacher is not None and state.get("teacher_sha256") not in {
            None,
            teacher.checkpoint_sha256,
        }:
            raise ValueError("Resume changes the frozen teacher weights")
        optimizer.load_state_dict(state["optimizer"])
        first_step, history = state["step"], state["history"]
        local_state = state["rank_states"][rank] if len(state["rank_states"]) == world else None
        if local_state is None:
            raise ValueError("Exact resume requires the same number of DDP ranks")
        epoch, offset = local_state["epoch"], local_state["offset"]
        loader_generator.set_state(local_state["epoch_generator"].cpu())
        resume_rng = local_state["rng"]
        if time_sampler and state["time_sampler"]:
            time_sampler.load_state_dict(state["time_sampler"])
    else:
        torch.manual_seed(cfg.seed + rank)
    if rank == 0 and first_step < cfg.steps:
        (directory / "training_complete.json").unlink(missing_ok=True)
    if sampler:
        sampler.set_epoch(epoch)
    epoch_generator = loader_generator.get_state()
    if resume_rng is not None and cfg.workers > 0 and epoch > 0:
        # Later persistent-worker epochs reuse their original base seed. A fresh
        # loader would consume a new seed before drawing the saved permutation.
        # Prime its workers, rewind the order RNG, then use the normal reset path.
        iter(loader)
        loader_generator.set_state(epoch_generator)
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    if resume_rng:
        restore_rng(resume_rng)

    def batch():
        nonlocal iterator, epoch, offset, epoch_generator
        try:
            clean, labels = next(iterator)
        except StopIteration:
            epoch, offset = epoch + 1, 0
            if sampler:
                sampler.set_epoch(epoch)
            epoch_generator = loader_generator.get_state()
            iterator = iter(loader)
            clean, labels = next(iterator)
        offset += 1
        clean = clean.to(device, non_blocking=True)
        if is_latent:
            clean = posterior_samples(clean, latent_manifest["normalization"])
        if cfg.data.get("horizontal_flip", False):
            flip = torch.rand(len(clean), device=device) < 0.5
            clean[flip] = clean[flip].flip(-1)
        return clean, labels.to(device, non_blocking=True)

    fixed_indices = balanced_image_indices(validation, min(128, len(validation)))
    fixed_clean = torch.stack([validation[i][0] for i in fixed_indices]).to(device)
    if is_latent:
        fixed_clean = posterior_samples(
            fixed_clean,
            latent_manifest["normalization"],
            torch.Generator(device=device).manual_seed(cfg.seed + 10001),
            augment=False,
        )
    fixed_labels = torch.tensor([validation[i][1] for i in fixed_indices], device=device)

    def objective(model, clean, labels, step, generator=None, adaptive=True):
        if cfg.algorithm == "edm":
            return edm_loss(model, clean, labels, cfg.p_mean, cfg.p_std, generator)
        if cfg.algorithm == "improved_consistency":
            scales = improved_scales(
                step, cfg.steps, cfg.consistency_scales_min, cfg.consistency_scales_max
            )
            return improved_consistency_loss(
                model, clean, labels, scales, cfg.sigma_max, cfg.p_mean, cfg.p_std, generator
            )
        if cfg.algorithm == "progressive":
            return progressive_loss(
                model, teacher, teacher_cfg, path, clean, labels, cfg.student_steps, generator
            )
        if cfg.algorithm in {"consistency", "distillation"}:
            scales = math.ceil(
                math.sqrt(
                    (step / cfg.steps)
                    * (cfg.consistency_scales_max**2 - cfg.consistency_scales_min**2)
                    + cfg.consistency_scales_min**2
                )
            )
            return consistency_loss(
                model,
                target_ema.model,
                clean,
                labels,
                scales,
                cfg.sigma_max,
                teacher,
                generator,
                weighting=cfg.get("consistency_weighting", "uniform"),
                loss_norm=cfg.get("consistency_loss_norm", "mse"),
            )
        noise = torch.randn(clean.shape, device=device, generator=generator)
        if cfg.algorithm == "flow":
            t = torch.rand(len(clean), device=device, generator=generator)
            a, s, da, ds = flow_coefficients(t, cfg.flow_path)
            noisy, target = (
                expand(a, clean) * clean + expand(s, clean) * noise,
                expand(da, clean) * clean + expand(ds, clean) * noise,
            )
            return (model(noisy, t, labels).float() - target).square().flatten(1).mean(1)
        indices, importance = (
            time_sampler.sample(len(clean), generator)
            if time_sampler and adaptive
            else (
                torch.randint(path.steps, (len(clean),), device=device, generator=generator),
                torch.ones(len(clean), device=device),
            )
        )
        t = (indices.float() + 1) / path.steps
        a, s = path.coefficients(t)
        noisy = expand(a, clean) * clean + expand(s, clean) * noise
        from .targets import training_target

        target, weight = training_target(clean, noise, a, s, cfg.prediction)
        losses = improved_loss(
            path,
            clean,
            noisy,
            t,
            target,
            model(noisy, t, labels),
            cfg.prediction,
            cfg.objective,
            cfg.hybrid_weight,
            weight,
            cfg.learned_variance,
        )
        if time_sampler and adaptive:
            # All ranks update with the same gathered observations.
            if world > 1:
                all_indices, all_losses = (
                    [torch.empty_like(indices) for _ in range(world)],
                    [torch.empty_like(losses) for _ in range(world)],
                )
                dist.all_gather(all_indices, indices)
                dist.all_gather(all_losses, losses.detach())
                time_sampler.update(torch.cat(all_indices), torch.cat(all_losses))
            else:
                time_sampler.update(indices, losses.detach())
        return losses * importance

    if rank == 0:
        OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
        save_json(
            directory / "environment.json",
            {
                "torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "world_size": world,
                "teacher_sha256": teacher.checkpoint_sha256 if teacher is not None else None,
                "parameters": sum(p.numel() for p in network.parameters()),
                "train_count": len(data),
                "validation_count": len(validation),
                "class_names": data.class_names,
                "representation": "whitened autoencoder posterior" if is_latent else "pixels",
                "latent_manifest_sha256": data.manifest_sha256 if is_latent else None,
            },
        )
    started, total, logged = time.perf_counter(), 0.0, 0
    for step in range(first_step + 1, cfg.steps + 1):
        train_model.train()
        warmup = min(1.0, step / max(1, cfg.warmup_steps))
        decay = (
            0.5 * (1 + math.cos(math.pi * step / cfg.steps)) if cfg.lr_schedule == "cosine" else 1
        )
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * warmup * decay
        optimizer.zero_grad(set_to_none=True)
        local_loss = 0.0
        for _ in range(cfg.accumulation):
            clean, labels = batch()
            labels = drop_labels(labels, classes, cfg.label_dropout) if classes else None
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=cfg.amp and device.type == "cuda",
            ):
                losses = objective(train_model, clean, labels, step)
                loss = losses.mean() / cfg.accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at step {step}")
            loss.backward()
            local_loss += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), cfg.grad_clip)
        optimizer.step()
        # Ramp EMA for a useful model early; asymptotic decay remains configured.
        ema.decay = min(cfg.ema_decay, (1 + step) / (10 + step))
        ema.update(network)
        if target_ema is not ema:
            target_ema.update(network)
        total += local_loss
        logged += 1
        if step % cfg.log_every == 0 or step == cfg.steps:
            row = {
                "step": step,
                "train_loss": total / logged,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
                "images_seen": step * cfg.batch_size * cfg.accumulation * world,
            }
            value = torch.tensor(row["train_loss"], device=device)
            if world > 1:
                dist.all_reduce(value)
            row["train_loss"] = value.item() / world
            if rank == 0:
                history.append(row)
                print(json.dumps(row), flush=True)
            total, logged = 0.0, 0
        if step % cfg.save_every == 0 or step == cfg.steps:
            local = {
                "rng": rng_state(),
                "epoch": epoch,
                "offset": offset,
                "epoch_generator": epoch_generator,
            }
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, local)
            else:
                states[0] = local
            if rank == 0:
                ema.model.eval()
                generator = torch.Generator(device=device).manual_seed(cfg.seed + 10000)
                with torch.no_grad():
                    value = (
                        objective(
                            ema.model,
                            fixed_clean,
                            fixed_labels if classes else None,
                            step,
                            generator,
                            adaptive=False,
                        )
                        .mean()
                        .item()
                    )
                history.append({"step": step, "validation_loss": value})
                payload = {
                    "format": "image_v2",
                    "model": network.state_dict(),
                    "ema": ema.model.state_dict(),
                    "target_ema": target_ema.model.state_dict() if target_ema is not ema else None,
                    "teacher_sha256": teacher.checkpoint_sha256 if teacher is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "classes": classes,
                    "class_names": data.class_names,
                    "step": step,
                    "history": history,
                    "rank_states": states,
                    "time_sampler": time_sampler.state_dict() if time_sampler else None,
                    "latent_manifest": latent_manifest,
                }
                atomic_checkpoint(directory / "checkpoint.pt", payload)
                save_json(directory / "history.json", history)
                loss_plot(history, directory / "loss.png")
                print(
                    json.dumps(
                        {
                            "step": step,
                            "validation_loss": value,
                            "checkpoint": str(directory / "checkpoint.pt"),
                        }
                    ),
                    flush=True,
                )
            if world > 1:
                dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
        if step % cfg.preview_every == 0 or step == cfg.steps:
            if rank == 0:
                generator = torch.Generator(device=device).manual_seed(cfg.seed + 20000)
                noise = torch.randn(
                    40,
                    cfg.data.channels,
                    cfg.data.image_size,
                    cfg.data.image_size,
                    device=device,
                    generator=generator,
                )
                labels = torch.arange(40, device=device) % classes if classes else None
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=cfg.amp and device.type == "cuda",
                ):
                    images, trajectory, nfe = sample_images(
                        cfg, ema.model, noise, labels, cfg.preview_scale, generator=generator
                    )
                if is_latent:
                    images = codec.decode(images, latent_manifest["normalization"])
                    trajectory = decode_trajectory(
                        codec, trajectory, latent_manifest["normalization"]
                    )
                save_image(
                    (images.cpu().clamp(-1, 1) + 1) / 2,
                    directory / f"preview_{step:07d}.png",
                    nrow=10,
                )
                image_artifacts(images.cpu(), trajectory, directory)
                save_json(
                    directory / "preview.json",
                    {
                        "step": step,
                        "nfe_per_image": nfe,
                        "cfg_scale": cfg.preview_scale,
                        "layout": "rows repeat class 0 through class 9"
                        if classes == 10
                        else "class labels cycle",
                    },
                )
            if world > 1:
                dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
    if rank == 0:
        save_json(
            directory / "training_complete.json",
            {
                "step": cfg.steps,
                "status": "configured training budget completed; quality requires evaluation",
            },
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
