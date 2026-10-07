"""COCO/custom caption training with our compact U-Net or local SD2 LoRA.

Frozen codecs/text encoders remain external. Checkpoints contain only trainable
denoiser weights, their EMA, optimizer, exact rank RNG and data-order position.
torchrun supports one process per GPU. No network access or uploads occur here.
"""

import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .caption_data import CaptionDataset, EpochSampler, caption_collate
from .text import TextUNet, inject_lora, load_trainable_state, trainable_state
from .train import CONFIG_ROOT
from .utils import atomic_checkpoint, save_json, seed_everything


def model_fingerprint(folder):
    """Hash configs, tokenizer and model weights; portable content identity."""
    folder = Path(folder)
    rows = {}

    def visit(directory, ancestors):
        real = directory.resolve()
        if real in ancestors:
            raise ValueError("Base model contains a cyclic directory symlink")
        for path in sorted(directory.iterdir()):
            if path.name.startswith("."):
                continue
            if path.is_dir():
                visit(path, ancestors | {real})
            elif path.is_file() and path.suffix in {".json", ".txt", ".safetensors", ".bin"}:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                rows[str(path.relative_to(folder))] = digest.hexdigest()

    visit(folder, set())
    if not rows:
        raise ValueError("Base model has no recognized local files")
    return rows


def codec_geometry(vae, resolution):
    """AutoencoderKL's spatial factor and supported scalar latent transform."""
    factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if resolution < factor or resolution % factor:
        raise ValueError(f"Image resolution must be divisible by codec factor {factor}")
    scale = float(vae.config.scaling_factor)
    shift = float(getattr(vae.config, "shift_factor", None) or 0)
    if scale <= 0 or not all(torch.isfinite(torch.tensor([scale, shift]))):
        raise ValueError("Invalid codec scale or shift")
    if any(getattr(vae.config, key, None) is not None for key in ("latents_mean", "latents_std")):
        raise ValueError("Per-channel latent normalization needs an explicit codec adapter")
    return factor, scale, shift


def encode_latents(vae, images):
    _, scale, shift = codec_geometry(vae, images.shape[-1])
    return (vae.encode(images).latent_dist.sample() - shift) * scale


def decode_latents(vae, latents):
    # Geometry is determined by the codec, not a hard-coded factor of eight.
    factor = 2 ** (len(vae.config.block_out_channels) - 1)
    _, scale, shift = codec_geometry(vae, latents.shape[-1] * factor)
    return vae.decode(latents.float() / scale + shift).sample


def load_components(cfg, device):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    base = str(Path(cfg.base_model).expanduser())
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", local_files_only=True
    ).to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", local_files_only=True).to(device)
    codec_geometry(vae, cfg.resolution)
    text_encoder.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler", local_files_only=True)
    if scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError("Supported scheduler targets are epsilon and v_prediction")
    if cfg.mode == "lora":
        dtype = torch.bfloat16 if cfg.amp and device.type == "cuda" else torch.float32
        model = UNet2DConditionModel.from_pretrained(
            base, subfolder="unet", local_files_only=True, torch_dtype=dtype
        ).to(device)
        inject_lora(model, cfg.rank, cfg.lora_alpha)
        if cfg.gradient_checkpointing:
            model.enable_gradient_checkpointing()
    elif cfg.mode == "compact":
        model = TextUNet(
            text_dim=text_encoder.config.hidden_size,
            width=cfg.width,
            channels=vae.config.latent_channels,
            channel_mult=cfg.channel_mult,
            blocks_per_level=cfg.blocks_per_level,
            attention_levels=(1, 2),
            dropout=0,
            time_embedding=cfg.get("time_embedding", "fourier"),
            time_scale=cfg.get("time_scale", 1.0),
        ).to(device)
    else:
        raise ValueError("mode must be lora or compact")
    model.diffusion_steps = scheduler.config.num_train_timesteps
    return model, vae, text_encoder, tokenizer, scheduler


@torch.no_grad()
def encode_text(text_encoder, tokenizer, captions, device):
    tokens = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    kwargs = (
        {"attention_mask": tokens.attention_mask}
        if getattr(text_encoder.config, "use_attention_mask", False)
        else {}
    )
    embeddings = text_encoder(tokens.input_ids, **kwargs).last_hidden_state
    return embeddings, tokens.attention_mask.bool()


def native_prediction(model, cfg, x, times, context):
    hidden, mask = context
    if cfg.mode == "compact":
        core = model.module if hasattr(model, "module") else model
        return model(x, (times.float() + 1) / core.diffusion_steps, hidden, mask)
    # Preserve pretrained SD2 semantics: its U-Net was conditioned on all CLIP
    # positions, including EOS/padding; changing that mask would change the model.
    return model(x, times, encoder_hidden_states=hidden).sample


def diffusion_training_loss(prediction, target, scheduler, times, gamma=5.0):
    per_example = (prediction.float() - target.float()).square().flatten(1).mean(1)
    if gamma is None:
        return per_example.mean()
    if gamma <= 0:
        raise ValueError("Min-SNR gamma must be positive or null")
    alpha_bar = scheduler.alphas_cumprod.to(times.device)[times]
    snr = alpha_bar / (1 - alpha_bar).clamp_min(1e-12)
    denominator = snr + (scheduler.config.prediction_type == "v_prediction")
    weight = torch.minimum(snr, torch.full_like(snr, gamma)) / denominator.clamp_min(1e-12)
    return (per_example * weight).mean()


@hydra.main(version_base=None, config_path=str(CONFIG_ROOT), config_name="text_train")
def main(cfg):
    if (
        min(
            cfg.steps,
            cfg.batch_size,
            cfg.accumulation,
            cfg.save_every,
            cfg.log_every,
            cfg.milestone_every,
            cfg.lr,
            cfg.grad_clip,
        )
        <= 0
    ):
        raise ValueError("Training counts and optimization scales must be positive")
    if not 0 < cfg.caption_dropout < 1 or not 0 <= cfg.ema_decay < 1 or cfg.workers < 0:
        raise ValueError("Invalid CFG dropout, EMA or worker count")
    world, rank, local_rank = (
        int(os.environ.get(k, d)) for k, d in [("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)]
    )
    device = torch.device(f"cuda:{local_rank}" if world > 1 else cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed_everything(cfg.seed, cfg.cpu_threads)
    directory = Path(to_absolute_path(cfg.run_dir))
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoint.pt").exists() and not cfg.resume:
        raise ValueError("Choose a new run directory or explicitly resume")
    data = CaptionDataset(
        cfg.data_root, resolution=cfg.resolution, limit=cfg.train_limit, manifest=cfg.train_manifest
    )
    identity = model_fingerprint(cfg.base_model) if rank == 0 else None
    if world > 1:
        identities = [identity]
        dist.broadcast_object_list(identities, 0)
        identity = identities[0]
    model, vae, text_encoder, tokenizer, scheduler = load_components(cfg, device)
    train_model = (
        DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
        if world > 1
        else model
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = {name: value.to(device) for name, value in trainable_state(model).items()}
    sampler = EpochSampler(data, cfg.seed, rank, world)
    loader = DataLoader(
        data,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.workers,
        persistent_workers=cfg.workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=caption_collate,
        generator=torch.Generator().manual_seed(cfg.seed + rank),
    )
    if not len(loader):
        raise ValueError("Per-rank dataset is smaller than batch size")
    start, epoch, offset, history = 0, 0, 0, []
    local_state = None
    identity_provenance = {"full_hashes_captured_at_step": 0, "initial_identity_complete": True}
    if cfg.resume:
        state = torch.load(to_absolute_path(cfg.resume), map_location="cpu", weights_only=False)
        current = OmegaConf.to_container(cfg, resolve=True)
        mutable = {
            "resume",
            "run_dir",
            "device",
            "save_every",
            "log_every",
            "milestone_every",
            "base_model",
            "data_root",
            "train_manifest",
            "steps",
        }
        compatibility = {"ema_warmup": False, "time_embedding": "fourier", "time_scale": 1.0}
        if any(
            current[k] != state["config"].get(k, compatibility.get(k))
            for k in current
            if k not in mutable
        ):
            raise ValueError("Resume changes the saved training configuration")
        if cfg.steps < state["step"]:
            raise ValueError("Resume budget is smaller than the saved step")
        if (
            state["base_identity"] != identity
            or state["manifest_sha256"] != data.manifest_sha256
            or len(state["rank_states"]) != world
        ):
            raise ValueError("Resume changes base weights, dataset manifest or rank count")
        load_trainable_state(model, state["model"])
        ema = {name: value.to(device) for name, value in state["ema"].items()}
        optimizer.load_state_dict(state["optimizer"])
        start, history = state["step"], state["history"]
        local_state = state["rank_states"][rank]
        epoch, offset = local_state["epoch"], local_state["offset"]
        identity_provenance = state.get("base_identity_provenance", identity_provenance)
    seed_everything(cfg.seed + rank, cfg.cpu_threads)
    sampler.epoch = epoch
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    if local_state:
        torch.set_rng_state(local_state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(local_state["cuda_rng"], device)
    amp = cfg.amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        empty = encode_text(text_encoder, tokenizer, [""] * cfg.batch_size, device)
    if rank == 0:
        OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
        save_json(
            directory / "environment.json",
            {
                "torch": torch.__version__,
                "world_size": world,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "base_identity": identity,
                "base_identity_provenance": identity_provenance,
                "manifest_sha256": data.manifest_sha256,
                "pretrained_components": [
                    "VAE",
                    "text encoder",
                    *(["SD2 denoiser"] if cfg.mode == "lora" else []),
                ],
                "prediction_type": scheduler.config.prediction_type,
                "effective_batch_size": world * cfg.batch_size * cfg.accumulation,
            },
        )
        if start < cfg.steps:
            (directory / "training_complete.json").unlink(missing_ok=True)
    started = time.perf_counter()
    model.train()
    for step in range(start + 1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        lr = cfg.lr * min(1.0, step / max(1, cfg.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        total = torch.zeros((), device=device)
        for micro in range(cfg.accumulation):
            try:
                images, candidates, _ids = next(iterator)
            except StopIteration:
                epoch, offset = epoch + 1, 0
                sampler.epoch = epoch
                iterator = iter(loader)
                images, candidates, _ids = next(iterator)
            offset += 1
            captions = [choices[int(torch.randint(len(choices), ()))] for choices in candidates]
            images = images.to(device, non_blocking=True)
            # This codec remains FP32 for stable posterior moments.
            with torch.no_grad():
                latents = encode_latents(vae, images)
            noise = torch.randn_like(latents)
            times = torch.randint(
                scheduler.config.num_train_timesteps, (len(latents),), device=device
            )
            noisy = scheduler.add_noise(latents, noise, times)
            target = (
                noise
                if scheduler.config.prediction_type == "epsilon"
                else scheduler.get_velocity(latents, noise, times)
            )
            synchronize = (
                train_model.no_sync()
                if world > 1 and micro < cfg.accumulation - 1
                else nullcontext()
            )
            with (
                synchronize,
                torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp),
            ):
                context = encode_text(text_encoder, tokenizer, captions, device)
                dropout = torch.rand(len(latents), device=device) < cfg.caption_dropout
                context = tuple(
                    torch.where(dropout.reshape(-1, *([1] * (a.ndim - 1))), b, a)
                    for a, b in zip(context, empty, strict=True)
                )
                prediction = native_prediction(train_model, cfg, noisy, times, context)
                loss = diffusion_training_loss(
                    prediction, target, scheduler, times, cfg.min_snr_gamma
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Text diffusion loss became nonfinite")
                (loss / cfg.accumulation).backward()
                total += loss.detach() / cfg.accumulation
        torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            decay = (
                min(cfg.ema_decay, (1 + step) / (10 + step))
                if cfg.get("ema_warmup", False)
                else cfg.ema_decay
            )
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    ema[name].lerp_(parameter.detach().float(), 1 - decay)
        if step % cfg.log_every == 0 or step == cfg.steps:
            if world > 1:
                dist.all_reduce(total)
                total /= world
            if rank == 0:
                row = {
                    "step": step,
                    "loss": total.item(),
                    "lr": lr,
                    "seconds_per_optimizer_step": (time.perf_counter() - started) / (step - start),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                save_json(directory / "history.json", history)
        if step % cfg.save_every == 0 or step == cfg.steps:
            local = {
                "epoch": epoch,
                "offset": offset,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device).cpu()
                if device.type == "cuda"
                else None,
            }
            ranks = [None] * world
            if world > 1:
                dist.all_gather_object(ranks, local)
            else:
                ranks = [local]
            if rank == 0:
                payload = {
                    "format": "text_v1",
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "step": step,
                    "model": trainable_state(model),
                    "ema": ema,
                    "optimizer": optimizer.state_dict(),
                    "rank_states": ranks,
                    "base_identity": identity,
                    "base_identity_provenance": identity_provenance,
                    "manifest_sha256": data.manifest_sha256,
                    "history": history,
                    "prediction_type": scheduler.config.prediction_type,
                }
                atomic_checkpoint(directory / "checkpoint.pt", payload)
                if step % cfg.milestone_every == 0 or step == cfg.steps:
                    atomic_checkpoint(directory / f"checkpoint_{step:07d}.pt", payload)
    if rank == 0:
        save_json(
            directory / "training_complete.json",
            {
                "step": cfg.steps,
                "mode": cfg.mode,
                "status": "Configured training budget completed; see separate held-out evaluation",
            },
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
