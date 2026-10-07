#!/bin/bash
#SBATCH --job-name=latent-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
: "${VAE_PATH:?Set VAE_PATH to a local AutoencoderKL directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -m diffusion_lab.latents --vae "$VAE_PATH" \
  --data "${DATASET:-celeba}" --image-size "${IMAGE_SIZE:-128}" \
  --output "${LATENT_CACHE_ROOT:-outputs/celeba128_latent_cache}" --device cuda:0 "$@"
