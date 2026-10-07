#!/bin/bash
#SBATCH --job-name=coco-text
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m diffusion_lab.text_train "$@"
