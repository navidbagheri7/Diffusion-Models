#!/bin/bash
#SBATCH --job-name=diffusion-dit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120

set -euo pipefail
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RECIPE="${RECIPE:-image_dit}"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m diffusion_lab.image_train --config-name "$RECIPE" "$@"
