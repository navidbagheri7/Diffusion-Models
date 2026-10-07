"""Audit the Git file set, local exclusions, size and limited credential patterns."""

import argparse
import json
import re
from pathlib import Path

if __package__:
    from ._publication import ignored_paths, publication_files, tracked_ignored_paths
else:
    from _publication import ignored_paths, publication_files, tracked_ignored_paths


IGNORE_PROBES = [
    "outputs/run/config.yaml",
    "multirun/run/.hydra/config.yaml",
    ".local/session_scripts/finish_image_runs.py",
    "configs/local/private.yaml",
    "legacy/src/unet.py",
    "data/example.jpg",
    "datasets/coco2017/train.jsonl",
    "checkpoints/weights.safetensors",
    "model.pt",
    "optimizer.pth",
    "model.ckpt",
    "pytorch_model.bin",
    "samples.npy",
    "posterior.npz",
    "train.zip",
    ".cache/hub/model.json",
    ".env",
    ".env.local",
    "private.pem",
    "process.log",
    "slurm-123.out",
    "slurm-123.err",
    "slurm/logs/coco-text-123.out",
    ".venv/pyvenv.cfg",
    ".ipynb_checkpoints/lesson.ipynb",
    "dist/package.whl",
    "build/lib/module.py",
]

REQUIRED_FILES = [
    "README.md",
    "CITATION.cff",
    "pyproject.toml",
    "diffusion_lab/train.py",
    "configs/train.yaml",
    "notebooks/00_probability_and_noising.ipynb",
    "assets/teaching/gaussian_diffusion/ode_sde_trajectories.png",
    "assets/experiments/cifar10/cifar10_edm_cfg_final/metrics.json",
    "assets/metadata/checkpoints.json",
    "legacy/README.md",
    "slurm/train_text.sbatch",
    "tests/test_math.py",
    ".github/workflows/tests.yml",
]

SECRET_PATTERNS = {
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "aws_access_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    "github_fine_grained_token": r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    "openai_token": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b",
    "huggingface_token": r"\bhf_[A-Za-z0-9]{30,}\b",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-file-mib", type=float, default=20)
    parser.add_argument("--report", type=Path, default=Path("outputs/publication_audit.json"))
    args = parser.parse_args()
    if args.max_file_mib <= 0:
        parser.error("--max-file-mib must be positive")
    root = Path(__file__).resolve().parents[1]
    files = publication_files(root)
    relative = {str(path.relative_to(root)) for path in files}
    ignored = ignored_paths(root, IGNORE_PROBES)
    missing = sorted(set(REQUIRED_FILES) - relative)
    exclusions_missing = sorted(set(IGNORE_PROBES) - ignored)
    symlinks = [str(path.relative_to(root)) for path in files if path.is_symlink()]
    oversized = [
        str(path.relative_to(root))
        for path in files
        if path.stat().st_size > args.max_file_mib * 2**20
    ]
    secret_matches = []
    for path in files:
        if path.suffix not in {
            ".py",
            ".md",
            ".json",
            ".ipynb",
            ".yaml",
            ".yml",
            ".toml",
            ".cff",
            ".txt",
            ".sbatch",
            ".sh",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, text):
                # Never include the credential value or surrounding text in reports.
                secret_matches.append({"file": str(path.relative_to(root)), "kind": name})
    tracked_ignored = [str(path) for path in tracked_ignored_paths(root)]
    problems = {
        "missing_public_files": missing,
        "missing_exclusions": exclusions_missing,
        "symlinks": symlinks,
        "oversized_files": oversized,
        "credential_pattern_matches": secret_matches,
        "already_tracked_ignored_files": tracked_ignored,
    }
    passed = not any(problems.values())
    largest = sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:10]
    report = {
        "passed": passed,
        "candidate_public_files": len(files),
        "candidate_source_assets_mib": sum(path.stat().st_size for path in files) / 2**20,
        "maximum_file_mib_policy": args.max_file_mib,
        "largest_files": [
            {"file": str(p.relative_to(root)), "mib": p.stat().st_size / 2**20} for p in largest
        ],
        "ignore_probes": {path: path in ignored for path in IGNORE_PROBES},
        "problems": problems,
        "scope": "Git ignore semantics, file presence/size, symlinks and limited credential patterns. Not a proof of scientific correctness, model quality, licensing or exhaustive secret detection.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Publication audit: {'passed' if passed else 'failed'}; {len(files)} files, {report['candidate_source_assets_mib']:.2f} MiB. Report: {args.report}"
    )
    if not passed:
        print(json.dumps(problems, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
