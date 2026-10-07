#!/bin/bash
#SBATCH --job-name=dsbm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -m diffusion_lab.dsbm --device cuda:0 "$@"
