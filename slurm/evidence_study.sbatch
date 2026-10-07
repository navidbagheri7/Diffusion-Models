#!/bin/bash
#SBATCH --job-name=diffusion-evidence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDY="${STUDY:-likelihood}"
exec "$PYTHON_BIN" -m scripts.run_evidence_studies --study "$STUDY" \
  --device cuda:0 --output "outputs/${STUDY}_slurm" "$@"
