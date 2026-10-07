"""Execute all course notebooks and save static outputs plus validation evidence."""

import argparse
import hashlib
import os
import platform
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from diffusion_lab.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("notebooks"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--notebook", action="append", help="Execute selected filenames only")
    parser.add_argument(
        "--report", type=Path, default=Path("assets/reports/diagnostics/notebook_validation.json")
    )
    args = parser.parse_args()
    results = []
    paths = sorted(args.folder.glob("*.ipynb"))
    if args.notebook:
        requested = set(args.notebook)
        paths = [path for path in paths if path.name in requested]
        if requested != {path.name for path in paths}:
            raise ValueError("Requested notebook does not exist in the selected folder")
    for path in paths:
        started = time.perf_counter()
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name=args.kernel,
            resources={"metadata": {"path": str(path.parent.resolve())}},
        )
        client.execute()
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if any(cell.execution_count is None for cell in code_cells):
            raise RuntimeError(f"Unexecuted code cells in {path}")
        if any(output.output_type == "error" for cell in code_cells for output in cell.outputs):
            raise RuntimeError(f"Error outputs in {path}")
        # Default lessons render static outputs; avoid stale widget state in Git.
        notebook.metadata.pop("widgets", None)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        result = {
            "notebook": str(path),
            "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
            "seconds": time.perf_counter() - started,
            "status": "passed",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "markdown_cells": sum(c.cell_type == "markdown" for c in notebook.cells),
            "visual_outputs": sum(
                any(key.startswith("image/") for key in output.get("data", {}))
                for cell in code_cells
                for output in cell.outputs
            ),
        }
        results.append(result)
        print(result, flush=True)
    if not results:
        raise ValueError("No notebooks found")
    save_json(
        args.report,
        {
            "notebooks": results,
            "all_executed": True,
            "scope": "Fresh-kernel CPU lessons, small training runs, equations/shape assertions and static visuals; optional local-codec execution is recorded separately. No image-scale training is launched.",
            "course_revision": "2026-10-06",
            "python": platform.python_version(),
            "selected_subset": bool(args.notebook) or args.folder != Path("notebooks"),
            "optional_codec_enabled": bool(
                os.environ.get("DIFFUSION_NOTEBOOK_VAE")
                and os.environ.get("DIFFUSION_NOTEBOOK_IMAGE")
                and any(path.name == "09_latent_diffusion_and_dit.ipynb" for path in paths)
            ),
            "optional_codec_device": os.environ.get("DIFFUSION_NOTEBOOK_DEVICE", "cpu"),
        },
    )


if __name__ == "__main__":
    main()
