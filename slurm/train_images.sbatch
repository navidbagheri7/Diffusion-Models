#!/bin/bash
#SBATCH --job-name=diffusion-images
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120

set -euo pipefail
# Run sbatch from the repository root; set the site partition/account at submission.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-cifar10}"
ALGORITHM="${ALGORITHM:-edm}"
RUN_DIR="${RUN_DIR:-outputs/${DATASET}_${ALGORITHM}_slurm}"
mkdir -p "$RUN_DIR"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m diffusion_lab.image_train "data=$DATASET" "algorithm=$ALGORITHM" \
  "run_dir=$RUN_DIR" "$@"
