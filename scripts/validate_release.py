"""Verify portable inference, current notebook outputs and exact explorer bindings."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np

from diffusion_lab.adversarial import METHODS, digest
from diffusion_lab.utils import save_json


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/explorer_manifest.json").read_text())
    browser = json.loads((root / "docs/explorer_browser_validation.json").read_text())
    assert manifest["status"] == "complete" and manifest["concepts"] == 11
    assert manifest["presets"] == 200 and manifest["media"] == 567
    assert browser["status"] == "passed"
    assert digest(root / "docs/explorer.html") == browser["html_sha256"] == manifest["html_sha256"]
    assert len(browser["checked_views"]) == 567
    assert len(browser["live_analytic_checks"]) == 36 and len(browser["mobile_concepts"]) == 11
    assert browser["comparison_records"] and browser["checked_metric_fields"] > 0
    assert not browser["javascript_errors"]
    for filename, value in manifest["bindings"].items():
        assert digest(root / filename) == value, filename
    notebooks = json.loads(
        (root / "assets/reports/diagnostics/notebook_validation.json").read_text()
    )
    assert notebooks["all_executed"] and not notebooks["selected_subset"]
    assert notebooks["default_kernel_matches_interpreter"]
    assert len(notebooks["notebooks"]) == 11
    cells = 0
    for row in notebooks["notebooks"]:
        path = root / row["notebook"]
        assert digest(path) == row["sha256"], path
        nb = nbformat.read(path, as_version=4)
        code = [cell for cell in nb.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code)
        assert all(output.output_type != "error" for cell in code for output in cell.outputs)
        assert len(code) == row["code_cells"] and row["visual_outputs"] > 0
        cells += len(code)
    commands = []
    with tempfile.TemporaryDirectory(prefix="diffusion-public-generators-") as directory:
        for method in METHODS:
            for seed in [42, 43, 44]:
                folder = root / "assets/adversarial" / f"{method}_{seed}"
                output = Path(directory) / f"{method}_{seed}.npz"
                subprocess.run(
                    [
                        sys.executable,
                        str(root / "scripts/generate_adversarial.py"),
                        "--model",
                        str(folder),
                        "--output",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                with (
                    np.load(output, allow_pickle=False) as fresh,
                    np.load(folder / "samples.npz", allow_pickle=False) as saved,
                ):
                    np.testing.assert_array_equal(fresh["latent"], saved["latent"])
                    np.testing.assert_array_equal(fresh["generated"], saved["generated"])
                commands.append({"model": f"{method}_{seed}", "independent_predictions": 4096})
    report = {
        "status": "passed",
        "concepts": 11,
        "presets": 200,
        "media": 567,
        "notebooks": 11,
        "executed_code_cells": cells,
        "fresh_generation_commands": commands,
        "browser_live_checks": 36,
        "browser_comparison_records": len(browser["comparison_records"]),
        "browser_checked_metric_fields": browser["checked_metric_fields"],
        "scope": "Source/media-bound explorer and current fresh-kernel outputs; fifteen separate CPU inference commands reproduce every public GAN prediction. audit_adversarial.py separately recomputes GAN metrics/selection. Historical image evidence, optional codec checks and full test/packaging results retain their separate documented scope.",
    }
    save_json(root / "docs/validation.json", report)
    print("Verified 11 lessons and 15 independent GAN generation commands")


if __name__ == "__main__":
    main()
