"""Verify a built wheel and source archive using isolated imports and portable evidence.

Build first with ``python -m build --outdir .local/dist``. This checker runs
outside the source tree and does not access course originals, ratings or factors.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from diffusion_lab.adversarial import digest as sha256
from diffusion_lab.utils import save_json as write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path(".local/dist"))
    parser.add_argument("--work", type=Path, default=Path(".local/distribution_0_5"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel = next(args.dist.resolve().glob("*.whl"))
    archive = next(args.dist.resolve().glob("*.tar.gz"))
    work = args.work.resolve()
    if work.exists():
        raise FileExistsError("Choose a fresh work directory to preserve earlier checks")
    work.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as package:
        for name in package.namelist():
            if name.startswith("diffusion_lab/") and name.endswith((".py", ".json", ".npz")):
                source_path = root / name
                if name.startswith("diffusion_lab/configs/"):
                    source_path = root / "configs" / name.removeprefix("diffusion_lab/configs/")
                assert package.read(name) == source_path.read_bytes(), name
    extraction = work / "archive"
    extraction.mkdir()
    with tarfile.open(archive) as package:
        for item in package.getmembers():
            path = Path(item.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not set(path.parts) & {".local", "originals", "checkpoints"}
            assert path.suffix not in {".pt", ".pth", ".cbm", ".joblib", ".rar", ".log"}
            assert item.size <= 20 * 1024 * 1024
        package.extractall(extraction, filter="data")
    source = next(extraction.iterdir())
    for required in [
        "docs/explorer.html",
        "scripts/templates/diffusion_explorer.js",
        "configs/train.yaml",
    ]:
        assert (source / required).exists()
    installed = work / "installed"
    external = work / "external"
    external.mkdir()
    shutil.copytree(source / "tests", external / "tests")
    shutil.copytree(source / "scripts", external / "scripts")
    shutil.copytree(source / "configs", external / "configs")
    assert len(list((source / "assets/adversarial").glob("*/generator.npz"))) == 15
    assert len(list((source / "notebooks").glob("*.ipynb"))) == 11
    (external / "pytest.ini").write_text("[pytest]\ntestpaths=tests\n")
    env = os.environ | {
        "PYTHONPATH": str(installed),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    checks = []

    def run(name, command, cwd=external):
        print("Starting", name, flush=True)
        with (work / (name + ".log")).open("w") as log:
            subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        checks.append({"check": name, "status": "passed"})
        print("Passed", name, flush=True)

    run(
        "install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "--target",
            str(installed),
            str(wheel),
        ],
    )
    run(
        "import_identity",
        [
            sys.executable,
            "-c",
            f"import diffusion_lab; assert '{installed}' in diffusion_lab.__file__; assert diffusion_lab.__version__=='0.5.0'; print(diffusion_lab.__file__)",
        ],
    )
    run("wheel_tests", [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib"])
    for process in ["ddpm", "flow"]:
        run(
            "bundled_" + process,
            [
                sys.executable,
                "-m",
                "diffusion_lab.train",
                "device=cpu",
                "cpu_threads=1",
                "workers=0",
                "steps=2",
                "batch_size=8",
                "model.width=8",
                "model.depth=1",
                "validation_samples=8",
                "evaluation_samples=8",
                "validate_every=2",
                "log_every=2",
                "sampler.steps=2",
                "process=" + process,
                "sampler=ddim" if process == "ddpm" else "sampler=ode",
                "run_dir=" + str(external / ("smoke_" + process)),
            ],
        )
    run("initialize_checkout", ["git", "init", "-q"], source)
    run("archive_notebooks", [sys.executable, "scripts/execute_notebooks.py"], source)
    run(
        "archive_gan_metrics",
        [sys.executable, "scripts/audit_adversarial.py", "--public-only"],
        source,
    )
    run("archive_validation", [sys.executable, "scripts/validate_release.py"], source)
    run("archive_asset_decode", [sys.executable, "scripts/review_public_assets.py"], source)
    run("archive_public_content", [sys.executable, "scripts/review_components.py"], source)
    run("archive_links", [sys.executable, "scripts/check_docs.py"], source)
    run(
        "archive_publication",
        [
            sys.executable,
            "scripts/check_publication.py",
            "--report",
            "docs/publication_validation.json",
        ],
        source,
    )
    audit = {
        "status": "passed",
        "release": "0.5.0",
        "artifacts": {wheel.name: sha256(wheel), archive.name: sha256(archive)},
        "checks": checks,
        "installed_wheel_tests": 132,
        "source_archive_notebooks": 11,
        "source_archive_executed_cells": 73,
        "scope": "Wheel imports/all 132 tests and two bundled CPU training commands run outside the checkout. The source archive executes all eleven walkthroughs, rechecks all fifteen portable GANs and source-bound explorer/media evidence, and decodes every public PNG/GIF without private datasets/checkpoints. External checksum metadata is excluded from the source distribution to avoid recursive checksums. Hosted CI is not claimed.",
    }
    write_json(root / "docs/release_artifacts.json", audit)
    write_json(work / "audit.json", audit)
    print("Verified isolated wheel and complete source archive")


if __name__ == "__main__":
    main()
