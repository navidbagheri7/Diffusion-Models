# Latent diffusion and diffusion transformers

Latent diffusion changes the space in which noise is added. DiT changes the
denoiser architecture. They can be combined or studied separately:

| Representation | U-Net | DiT |
|---|---|---|
| Pixels | Existing CIFAR/MNIST/face generators | New class-conditioned CIFAR DiT |
| Frozen autoencoder latents | New 128px face latent U-Net | New 128px face latent DiT |

The three new image runs use EDM preconditioning. The DiT architecture follows
adaLN-Zero; its EDM objective and time encoding are this toolkit's recipe.
These experiments do not reproduce the original DiT paper's ImageNet setup or
scores. The generative denoisers are trained from scratch.

## What the autoencoder does

A frozen KL autoencoder gives posterior
$q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\mathrm{diag}(s_\phi(x)^2))$.
The generator learns a distribution of posterior latent draws, then the frozen
decoder turns generated latents into images. This reduces spatial work but
limits output detail to what the codec can reconstruct.

[latents.py](../diffusion_lab/latents.py) loads a local Diffusers AutoencoderKL
folder, freezes it, and keeps encoding/decoding in float32. Its manifest records
weight/config SHA-256, scale, shift, latent channels and downsampling factor.
Only the VAE is pretrained; no Stable Diffusion denoiser, transformer, text
encoder or pretrained image sampler is used.

The verified local SD3 autoencoder has 16 latent channels and an 8× spatial
reduction: a 128×128 RGB image becomes a 16×16×16 tensor. It uses
$z_s=(z-\mathrm{shift})\mathrm{scale}$, with scale 1.5305 and shift 0.0609.
The adapter reads these values from the actual config rather than assuming
another VAE's familiar scale. Compatible relocated weights must match hashes.
The VAE dependency is external and its weights are not bundled with the project.

## Reproducible posterior caching

The cache stores posterior **means and standard deviations**, not one fixed
random latent per image. Each training visit selects a pixel augmentation
variant and draws fresh Gaussian posterior noise using checkpointed trainer
RNG. Horizontal flips are encoded from flipped pixels; flipping a latent tensor
is not assumed equivalent to encoding a flipped image. The held-out split
uses its unaugmented variant.

Training-only channel moments define whitening:

$$m=\mathbb E_{train}[\mu_s],\qquad
v=\mathbb E_{train}[\mu_s^2+s_s^2]-m^2,\qquad
z_w=(z_s-m)/\sqrt v.$$

This includes posterior variance. Statistics computed from posterior means
alone underestimate the training latent variance. The inverse is
$z=z_w\sqrt v+m$, followed by $z/\mathrm{scale}+\mathrm{shift}$ before decoding.
Tests verify the inverse and empirical posterior variance.

Memory-mapped arrays make training independent of repeated expensive encoding.
The manifest records source ordering, split counts, cache-array hashes and
class names. Source ordering hashes cover filenames/sizes; they are not full
pixel-content hashes. Loading validates array shapes and hashes by default.
Each generator checkpoint embeds the representation manifest; resume rejects
a changed cache/codec/normalization. Encoder pretraining data are external, so
our train/held-out split does not establish that the VAE never saw these images.

~~~bash
pip install -e '.[latent,metrics]'
export VAE_PATH=/path/to/local/AutoencoderKL
python -m diffusion_lab.latents --vae "$VAE_PATH" --data celeba \
  --image-size 128 --device cuda:0 --workers 4 \
  --output outputs/celeba128_latent_cache
export LATENT_CACHE_ROOT=outputs/celeba128_latent_cache
python -m diffusion_lab.image_train --config-name image_latent_unet device=cuda:0
python -m diffusion_lab.image_train --config-name image_latent_dit device=cuda:1
~~~

For another cache set `data.channels` and `data.image_size` to its manifest's
latent shape, update `data.name`, and enable conditioning only if the cache
contains meaningful classes. The cache builder accepts a data-root override.
Custom image-folder split rules are described in [image practice](12_image_experiments.md).
The models in this chapter are class-conditioned/unconditional latent generation.
The separate [text-conditioning module](24_text_conditioning.md) now implements
token cross-attention and COCO adapter fine-tuning; loading the VAE here does not
turn these existing class/unconditional checkpoints into text-conditioned models.

## The DiT denoiser

[dit.py](../diffusion_lab/dit.py) patchifies an image or latent tensor with a
strided convolution. Tokens receive fixed separable 2D sinusoidal positions.
The shared conditioning module produces sinusoidal time features and optional
class/null embeddings. Each transformer block has LayerNorm, multihead scaled
dot-product attention, an MLP and two residual connections.

Time/class conditioning supplies six vectors: shift, scale and residual gate
for attention and for the MLP. For each branch,

$$h' = h+g(c)\odot F((1+s(c))\odot\mathrm{LN}(h)+b(c)).$$

Conditioning projections are initialized to zero, making blocks start as
identity maps. The final conditioned projection also starts at zero and
unpatchifies tokens into a dense output. This is adaLN-Zero, not removal of
time or class conditioning. Tests check identity initialization, output layout,
eventual class-embedding gradients and checkpointed-backward behavior.

The full runs use width 384, depth 12, six heads, MLP ratio 4 and patch size 2.
CIFAR-32 gives 256 tokens; 16×16 latents give 64 tokens. Reducing patch size
increases attention cost approximately quadratically in token count. Depth,
width, patch size and optional gradient checkpointing are Hydra parameters.
EDM passes $\log\sigma/4$ as time; discrete DDPM passes normalized corruption
time. The backbone also supports a doubled output for learned diagonal variance.

~~~bash
python -m diffusion_lab.image_train --config-name image_dit device=cuda:0
# Smaller architecture for a quick executable demonstration:
python -m diffusion_lab.image_train --config-name image_dit \
  model.width=128 model.depth=4 model.heads=4 steps=1000 \
  device=cuda:0 run_dir=outputs/cifar10_dit_demo
~~~

## Evaluation and portable execution

Latent evaluation decodes generated samples and measures genuine RGB FID/KID/IS
against the original held-out **pixel images**. Comparing generated latent
features against cached latents is not image FID. An optional reconstruction
baseline decodes held-out posterior means and reports PSNR and reconstruction
FID with the same references.

~~~bash
python -m diffusion_lab.image_evaluate \
  --checkpoint outputs/celeba128_latent_dit_quality/checkpoint.pt \
  --samples 10000 --steps 18 --scale 1 --device cuda:0 \
  --interpolation --reconstruction-baseline --output outputs/celeba128_latent_dit_final
# On another machine, preserve hashes while relocating codec and image data:
# add --vae /new/path/to/vae --data-root /new/path/to/images
~~~

Results retain generated latents, decoded tensors, noise-to-image GIFs,
SLERP interpolation strips/GIFs, configurations and actual denoiser NFE.
Decoder calls are excluded from denoiser NFE; recorded sampling time includes
decoding. Pixel noisy classifiers are not silently reused on latent noise.
No image likelihood is claimed from a latent log density alone.

Use [three-GPU DiT Slurm](../slurm/train_dit.sbatch) with
`RECIPE=image_latent_dit` or `image_dit`; build caches with
[the cache job](../slurm/cache_latents.sbatch). The trainer supports DDP, but
the new local experiments run independently on three GPUs to compare models.
Budgets and actual completed evidence appear in [extension results](21_extension_results.md).

References: [latent diffusion](https://arxiv.org/abs/2112.10752),
[DiT](https://arxiv.org/abs/2212.09748).
