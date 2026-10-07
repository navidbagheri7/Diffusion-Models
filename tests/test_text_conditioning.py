"""Conditioning invariants, frozen-base LoRA, data split isolation and CFG algebra."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from diffusion_lab.caption_data import CaptionDataset, EpochSampler
from diffusion_lab.consistency import ConsistencyModel, consistency_sample
from diffusion_lab.text import LowRankLinear, TextCrossAttention, TextUNet, text_guided_prediction
from diffusion_lab.text_train import (
    codec_geometry,
    decode_latents,
    diffusion_training_loss,
    encode_latents,
    model_fingerprint,
)


def test_cross_attention_masks_padding_and_normalizes():
    torch.manual_seed(1)
    model = TextCrossAttention(8, 6, heads=2)
    images = torch.randn(2, 8, 3, 3)
    text = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    first, weights = model(images, text, mask, return_weights=True)
    changed = text.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000
    torch.testing.assert_close(first, model(images, changed, mask))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 3))
    assert (weights[~mask[:, None, None, :].expand_as(weights)] == 0).all()
    with pytest.raises(ValueError):
        model(images, text, torch.zeros_like(mask))
    loss = first.square().mean()
    loss.backward()
    assert model.key.weight.grad.abs().sum() > 0


def test_lora_identity_then_updates_without_changing_base():
    torch.manual_seed(2)
    base = nn.Linear(6, 4)
    layer = LowRankLinear(base, rank=2, alpha=2)
    weights = base.weight.detach().clone()
    x = torch.randn(5, 6)
    torch.testing.assert_close(layer(x), base(x), atol=0, rtol=0)
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert layer.b.weight.grad.abs().sum() > 0
    optimizer.step()
    torch.testing.assert_close(base.weight, weights, atol=0, rtol=0)
    assert not torch.equal(layer(x), base(x))


def test_text_cfg_branch_counts_and_empty_prompt():
    calls = []

    def predict(x, t, context):
        calls.append(context)
        return x * 0 + context

    x = torch.zeros(2, 3)
    result, nfe = text_guided_prediction(predict, x, None, 3.0, 1.0, 2.0)
    torch.testing.assert_close(result, torch.full_like(x, 5))
    assert nfe == 2 and calls == [1.0, 3.0]
    result, nfe = text_guided_prediction(predict, x, None, 3.0, 1.0, 0.0)
    assert nfe == 1 and result.mean() == 1


def test_caption_manifest_isolation_and_safe_paths(tmp_path):
    image = Image.new("RGB", (12, 10), color="red")
    image.save(tmp_path / "example.png")
    rows = [
        {"image_id": i, "image": "example.png", "captions": ["a red square", "a red image"]}
        for i in range(1002)
    ]
    (tmp_path / "val.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    validation = CaptionDataset(tmp_path, "validation", resolution=8)
    evaluation = CaptionDataset(tmp_path, "evaluation", resolution=8)
    assert len(validation) == 1000 and len(evaluation) == 2
    assert {r["image_id"] for r in validation.rows}.isdisjoint(
        {r["image_id"] for r in evaluation.rows}
    )
    assert validation[0][0].shape == (3, 8, 8)
    rows[0]["image"] = "../outside.png"
    (tmp_path / "train.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError):
        CaptionDataset(tmp_path, resolution=8)


def test_epoch_sampler_rank_independence_and_reproducible_order():
    a, b = EpochSampler(range(20), 7, 0, 2), EpochSampler(range(20), 7, 1, 2)
    assert set(a).isdisjoint(set(b))
    initial = list(a)
    a.epoch = 1
    assert initial != list(a)
    a.epoch = 0
    assert initial == list(a)


def test_min_snr_target_specific_weights():
    x = torch.ones(2, 1)
    times = torch.tensor([0, 1])
    scheduler = SimpleNamespace(
        alphas_cumprod=torch.tensor([0.5, 0.9]), config=SimpleNamespace(prediction_type="epsilon")
    )
    torch.testing.assert_close(
        diffusion_training_loss(x, x * 0, scheduler, times, 5), torch.tensor((1 + 5 / 9) / 2)
    )
    scheduler.config.prediction_type = "v_prediction"
    torch.testing.assert_close(
        diffusion_training_loss(x, x * 0, scheduler, times, 5), torch.tensor((0.5 + 0.5) / 2)
    )


def test_compact_text_unet_trains_and_uses_token_context():
    torch.manual_seed(3)
    model = TextUNet(
        text_dim=6,
        width=8,
        channels=4,
        channel_mult=(1, 2),
        blocks_per_level=1,
        attention_levels=(1,),
        dropout=0,
    )
    x = torch.randn(2, 4, 8, 8)
    context = torch.randn(2, 4, 6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    target = torch.randn_like(x)
    (model(x, torch.ones(2) * 0.5, context) - target).square().mean().backward()
    optimizer.step()
    assert not torch.equal(
        model(x, torch.ones(2) * 0.5, context), model(x, torch.ones(2) * 0.5, context.flip(0))
    )


def test_explicit_consistency_levels_boundary_and_invalid_order():
    class Zero(nn.Module):
        def forward(self, x, t, labels=None):
            return torch.zeros_like(x)

    model = ConsistencyModel(Zero())
    initial = torch.zeros(4, 2, dtype=torch.float64)
    output, history, calls = consistency_sample(
        model, initial, steps=2, sigmas=[80, 1], generator=torch.Generator().manual_seed(2)
    )
    assert output.shape == initial.shape and len(history) == 3 and calls == 2
    with pytest.raises(ValueError):
        consistency_sample(model, initial, steps=2, sigmas=[80, 0.002])
    with pytest.raises(ValueError):
        consistency_sample(model, initial, steps=2, sigmas=[1, 80])


def test_codec_scale_shift_and_geometry_roundtrip():
    class Codec:
        config = SimpleNamespace(
            block_out_channels=[8, 16, 16],
            scaling_factor=0.5,
            shift_factor=0.25,
            latents_mean=None,
        )

        def encode(self, images):
            return SimpleNamespace(
                latent_dist=SimpleNamespace(sample=lambda: images[:, :, ::4, ::4])
            )

        def decode(self, values):
            return SimpleNamespace(sample=values)

    codec = Codec()
    image = torch.randn(2, 3, 16, 16)
    assert codec_geometry(codec, 16) == (4, 0.5, 0.25)
    latents = encode_latents(codec, image)
    torch.testing.assert_close(decode_latents(codec, latents), image[:, :, ::4, ::4])
    with pytest.raises(ValueError):
        codec_geometry(codec, 15)


def test_model_fingerprint_follows_component_symlinks_and_rejects_cycles(tmp_path):
    root = tmp_path / "base"
    component = tmp_path / "component"
    root.mkdir()
    component.mkdir()
    (component / "weights.bin").write_bytes(b"original weights")
    (root / "unet").symlink_to(component, target_is_directory=True)
    initial = model_fingerprint(root)
    assert "unet/weights.bin" in initial
    (component / "weights.bin").write_bytes(b"changed weights")
    assert initial != model_fingerprint(root)
    (component / "cycle").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="cyclic"):
        model_fingerprint(root)
