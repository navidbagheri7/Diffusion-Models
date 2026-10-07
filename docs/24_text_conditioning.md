# Text conditioning: captions, cross-attention, latent diffusion and LoRA

Text-conditioned generation is learned from paired images and descriptions.
Our class-conditioned checkpoints are not text-to-image models. This chapter
adds two explicit paths: our compact U-Net trained from scratch, and adapter
fine-tuning of a local pretrained SD2 denoiser. Both use an external frozen VAE
and frozen CLIP text encoder. The [latent diffusion paper](https://arxiv.org/abs/2112.10752)
motivates token conditioning; this implementation does not claim to reproduce
its large-scale training.

## Data and split protocol

```bash
python -m pip install -e ".[text,metrics]"
python -m scripts.download_coco --root /data/tii/Image_data/coco2017
```

The downloader uses the official COCO 2017 bucket, verifies HTTP ranges and
archive size, reads extracted members to verify CRC, and decodes every image.
It records local SHA-256 hashes without claiming upstream SHA-256 verification.
Train has 118,287 images; official val has 5,000. All available captions of an
image remain together in portable `train.jsonl`/`val.jsonl` manifests.
See [COCO](https://cocodataset.org/#download) for source and dataset terms.

Sorted val image IDs are split into the first 1,000 for model selection and
remaining 4,000 for final evaluation. This isolates our fine-tuning split;
external SD2/CLIP/VAE pretraining overlap is unknown. Deterministic center crops
can remove captioned objects near the border. Horizontal flips are disabled to
preserve left/right language; any future spatial augmentation must update text.

Custom data use one JSONL row per unique image ID:

```json
{"image_id": 1, "image": "images/example.jpg", "captions": ["a red bus beside a tree"]}
```

Pass `data_root=/your/root train_manifest=/your/train.jsonl`. Image paths must
resolve inside the root. Use a separate held-out custom manifest rather than
assuming COCO's 1k/4k split applies to your data.

## Cross-attention and shapes

Image features are `[B,C,H,W]`; text token embeddings are `[B,L,D]`. Flatten
spatial image features into `HW` queries. Project token features into keys and
values, then compute

$$A=\operatorname{softmax}(QK^\top/\sqrt d),\qquad O=AV.$$

Each location can combine information from several caption tokens. Our
[TextCrossAttention](../diffusion_lab/text.py) exposes optional attention weights
and validates token masks. It is used throughout our residual/time-conditioned
`TextUNet`. The [interactive notebook](../notebooks/06_text_cross_attention.ipynb)
demonstrates dimensions, masked padding and token influence; its random weights
are a mechanics lesson, not a trained semantic attention map.

The pretrained path preserves SD2's existing attention and padding conventions.
Loading it is explicitly recorded as pretrained-denoiser fine-tuning. Our native
LoRA layer freezes each original attention projection $W$ and trains

$$Wx+(\alpha_{\rm LoRA}/r)BAx.$$

$B$ starts at zero, so injection initially reproduces the base model. Adapter
weights and optimizer states are stored locally; the pretrained base is not
silently embedded or uploaded. The saved fingerprint must match on reload.
The scan follows cached component-directory symlinks and rejects cycles.
The COCO run's initial scan missed those directories; its audit records full
hashes captured at the step-1,750 continuation, which are enforced thereafter.
That provenance is retained rather than described as a full initial-run hash.
See [Diffusers LoRA](https://huggingface.co/docs/diffusers/v0.36.0/en/training/lora)
for the reference approach and conventions.

## Objective and classifier-free guidance

Encode images to latent posterior samples and apply the codec's scaling factor.
The base scheduler supplies the beta table and native prediction target; our
local SD2 checkpoint uses epsilon prediction. The text encoder provides token
features. MSE supervises epsilon or VP-v as declared by the scheduler.

Optional Min-SNR weights are $\min(\mathrm{SNR},\gamma)/\mathrm{SNR}$ for epsilon
and $\min(\mathrm{SNR},\gamma)/(\mathrm{SNR}+1)$ for VP-v. Setting
`min_snr_gamma=null` removes this weighting. This follows the target-specific
form of [Min-SNR](https://arxiv.org/abs/2303.09556), not an interchangeable weight
for every parameterization.

Caption dropout replaces conditioning with the encoded empty prompt. Sampling
uses $y_u+w(y_c-y_u)$: `scale=1` is ordinary conditioning, `scale=0` is empty
conditioning, and other positive values use two denoiser branches. Both weights
and intermediate states remain in the scheduler's native coordinates.

## Training and reproduction

```bash
# Point to a local Diffusers SD2 folder with unet/vae/text_encoder/tokenizer/scheduler.
export TEXT_BASE_MODEL=/path/to/local/sd2
python -m diffusion_lab.text_train
# Our denoiser from scratch; the codec and text encoder remain pretrained.
python -m diffusion_lab.text_train --config-name text_compact
# Larger random-denoiser COCO recipe with scaled positional time embeddings:
torchrun --standalone --nproc_per_node=2 -m diffusion_lab.text_train \
  --config-name text_coco_scratch
# Multiple GPUs, one process per GPU:
torchrun --standalone --nproc_per_node=3 -m diffusion_lab.text_train
# Resume with compatible hyperparameters and the same rank count:
python -m diffusion_lab.text_train resume=outputs/coco2017_sd2_lora/checkpoint.pt
```

Resolved Hydra configs, base-file hashes, dataset-manifest hash, optimizer, EMA,
rank RNG, epoch and consumed batch offset are retained. Captions are selected in
the main process with checkpointed RNG; deterministic workers cannot alter this
selection during resume. BF16 is optional, frozen codec moments remain FP32,
and the learning rate has linear warmup. A short benchmark determines throughput
before committing to a budget. [Slurm](../slurm/README.md) scripts provide the
same configuration entry points without hard-coding a cluster partition.
Paths can relocate when base-weight and caption-manifest identities match. The constant-after-warmup
schedule allows extending `steps` on resume; a smaller-than-saved budget is rejected.
For custom evaluation, pass `--data-root` and a separate `--manifest` JSONL.
Resume checks restore RNG and data position. The recorded CPU check differs
from uninterrupted weights by at most about $2.1\times10^{-8}$, and the short
BF16 CUDA check differs by about $1.9\times10^{-4}$. These are observed checks,
not a promise of bitwise identical optimization; see [resume evidence](../assets/reports/diagnostics/text_resume.json).

## Evaluation

```bash
python -m diffusion_lab.text_evaluate \
  --checkpoint outputs/coco2017_sd2_lora/checkpoint.pt \
  --output outputs/coco_text_eval --samples 256 --steps 30 --scale 5 \
  --clip /path/to/local/independent/clip
# Repeat identical prompts, settings and batch size with --base-only.
# For 4k/4k Inception diagnostics, use --samples 4000 --fid.
```

Evaluation saves rendered samples, raw tensors, exact captions/image IDs,
checkpoint hash, batch seeds and denoiser branch calls. An independent OpenAI
CLIP assessor reports paired cosine and a cyclically mismatched-caption baseline.
These measure broad semantic alignment; counting, binding and spatial relations
need targeted prompt comparisons and visual inspection. Custom prompt JSON lists
are supported through `--prompts`.

For controlled contrasts, pass `--matched-noise` so every prompt uses the same
initial latent. A saved gallery can then be assessed without regenerating it:

```bash
# The supplied JSON identifies the first three prompt pairs.
python -m scripts.evaluate_prompt_pairs --run outputs/coco_text_lora_pairs \
  --pairs assets/studies/prompts/contrast_pairs.json --clip /path/to/local/independent/clip
```

The color, count and fruit contrasts test correct versus alternative-caption
preference using CLIP. The last two saved prompts are paraphrases, so they are
excluded from the contrast accuracy. These six preference decisions are a
small teaching diagnostic, not a calibrated object-counting benchmark.
After the base/adapter and five CFG cases finish (0, 1, 3, 5 and 7.5), run
`python -m scripts.summarize_text_evaluation` to report matched per-caption
differences and caption-wise intervals. These intervals do not measure
uncertainty across fine-tuning seeds.

Always show the pretrained baseline alongside the adapter result using matched
noise/prompts. Report whether online or EMA weights were sampled. A visually
good pretrained output alone is not evidence that fine-tuning improved anything.
The experiment report distinguishes integration checks from trained results.

The `text_coco_scratch` recipe trains all 73.8M denoiser parameters from random
initialization. It uses 256px images (4×32×32 latent states), width 128,
multipliers [1,2,4], residual FiLM blocks, self-attention and token
cross-attention. Its positional embedding sees `1000*t`, where the trainer
passes `t=(discrete_index+1)/1000`. The empty-caption dropout remains 0.1.
The inference EMA warms up to its configured decay, making early checkpoints
useful without averaging predominantly untrained initial weights.

Alongside generated samples, test whether held-out captions help denoising:

```bash
python -m scripts.evaluate_text_denoising \
  --checkpoint outputs/coco2017_scratch_denoiser/checkpoint.pt \
  --output outputs/coco_scratch_denoising --device cuda:0
```

For each held-out image, the latent posterior draw and noise are shared across
the paired, empty and cyclically shuffled captions. The report gives prediction
MSE and paired differences at six discrete time indices. Positive differences
support useful conditioning; they do not establish good generated images.
The caption shuffle is a coarse control and can occasionally describe the
same semantic content. Intervals summarize variation across examples, not
independent training seeds. See the [training study](28_scratch_and_refinement.md)
for budgets, generation evaluation and the pretrained-component boundary.
