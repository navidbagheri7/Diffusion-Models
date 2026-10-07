"""Build the guided course from reviewable Python cell sources.

Regeneration clears execution state. Execute notebooks before publication so
the figures and results belong to the current source.
"""

import os
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

import nbformat

if __package__:
    from .notebook_lessons import LESSONS, SETUP
else:
    from notebook_lessons import LESSONS, SETUP

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    for lesson in LESSONS:
        introduction = (
            f"# {lesson['title']}\n\n{lesson['intro']}\n\n"
            f"**Prerequisites:** {lesson['prerequisites']}\n\n"
            f"**By the end:** {lesson['outcomes']}\n\n"
            "Run from top to bottom. Default experiments use CPU, fixed seeds and no downloads. "
            "Small trained models illustrate learning; analytic oracles isolate mathematical and solver effects. "
            "Neither is an image-quality benchmark.\n\n"
            "[Course index](README.md) · [Notation](../docs/23_notation_and_code.md) · "
            "[Implementation scope](../docs/15_coverage_and_scope.md)"
        )
        sources = [("md", introduction), ("code", SETUP), *lesson["cells"]]
        cells = []
        for index, (kind, source) in enumerate(sources):
            source = dedent(source).strip()
            cell = (
                nbformat.v4.new_markdown_cell(source)
                if kind == "md"
                else nbformat.v4.new_code_cell(source)
            )
            cell.id = sha256(f"{lesson['name']}:{index}:{source}".encode()).hexdigest()[:12]
            cells.append(cell)
        notebook = nbformat.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10"},
                "course": {"source": "scripts/notebook_lessons.py", "default_device": "cpu"},
            },
        )
        nbformat.write(notebook, folder / lesson["name"])
    formatter = os.environ.get("RUFF_BIN") or shutil.which("ruff")
    if not formatter:
        raise SystemExit("Install the dev extra (Ruff), or set RUFF_BIN, before building notebooks")
    subprocess.run([formatter, "format", str(folder)], check=True)
    subprocess.run([formatter, "check", "--fix", str(folder)], check=True)
    subprocess.run([formatter, "format", str(folder)], check=True)
    print(f"Built {len(LESSONS)} lessons. Execute them before publication.")


if __name__ == "__main__":
    main()
