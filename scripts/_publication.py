"""Find publication candidates using Git's actual ignore semantics."""

import subprocess
import tempfile
from pathlib import Path


def git_output(root, *arguments, git_dir=None, input_bytes=None):
    command = ["git", "-c", "core.excludesFile=/dev/null"]
    if git_dir is not None:
        command += ["--git-dir", str(git_dir), "--work-tree", str(root)]
    command += list(arguments)
    result = subprocess.run(command, cwd=root, input=input_bytes, capture_output=True, check=True)
    return result.stdout


def split_paths(value):
    return [Path(path.decode("utf-8")) for path in value.split(b"\0") if path]


def publication_files(root):
    """List files a new Git checkout would include, without initializing this repo.

    A temporary bare Git directory supplies an empty index. Global ignore rules
    are disabled, so every exclusion comes from this project's ignore files.
    """
    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix="diffusion-publication-") as temporary:
        git_dir = Path(temporary) / "index.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(git_dir)],
            cwd=root,
            capture_output=True,
            check=True,
        )
        paths = split_paths(
            git_output(root, "ls-files", "--others", "--exclude-standard", "-z", git_dir=git_dir)
        )
    return [root / path for path in sorted(paths)]


def ignored_paths(root, paths):
    """Check representative paths, including paths which do not exist yet."""
    with tempfile.TemporaryDirectory(prefix="diffusion-ignore-") as temporary:
        git_dir = Path(temporary) / "index.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(git_dir)],
            cwd=root,
            capture_output=True,
            check=True,
        )
        command = [
            "git",
            "-c",
            "core.excludesFile=/dev/null",
            "--git-dir",
            str(git_dir),
            "--work-tree",
            str(root),
            "check-ignore",
            "--no-index",
            "--stdin",
            "-z",
        ]
        result = subprocess.run(
            command,
            cwd=root,
            input=b"\0".join(str(path).encode() for path in paths) + b"\0",
            capture_output=True,
        )
        if result.returncode not in {0, 1}:
            raise RuntimeError(result.stderr.decode())
        return {str(path) for path in split_paths(result.stdout)}


def tracked_ignored_paths(root):
    """Ignoring a file already in Git's index does not untrack it."""
    if not (root / ".git").exists():
        return []
    return split_paths(
        git_output(root, "ls-files", "--cached", "--ignored", "--exclude-standard", "-z")
    )
