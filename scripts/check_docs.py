"""Check that documentation targets exist and are included by Git ignore rules."""

import re
from pathlib import Path

if __package__:
    from ._publication import publication_files
else:
    from _publication import publication_files


def main():
    root = Path(__file__).resolve().parents[1]
    missing = []
    count = 0
    public = publication_files(root)
    included = {path.resolve() for path in public}
    sources = [(path, path.read_text()) for path in public if path.suffix == ".md"]
    # Read only Markdown cells; output images and widget state are not prose links.
    import json

    for path in public:
        if path.suffix != ".ipynb":
            continue
        notebook = json.loads(path.read_text())
        sources.append(
            (
                path,
                "\n".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "markdown"
                ),
            )
        )
    for path, content in sources:
        # Brackets followed by parentheses also occur in LaTeX products.
        content = re.sub(r"\$\$.*?\$\$|\$[^\n]*?\$", "", content, flags=re.DOTALL)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#")[0]
            count += 1
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(root)} -> {target}")
            elif resolved.is_file() and resolved not in included:
                missing.append(f"{path.relative_to(root)} -> {target} (excluded from publication)")
            elif resolved.is_dir() and not any(p.is_relative_to(resolved) for p in included):
                missing.append(f"{path.relative_to(root)} -> {target} (no included files)")
    if missing:
        raise SystemExit("\n".join(missing))
    print(f"All {count} local Markdown targets exist and are included for publication.")


if __name__ == "__main__":
    main()
