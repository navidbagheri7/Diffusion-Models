"""Include the single source of Hydra configurations in built wheels."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithConfigs(build_py):
    def run(self):
        super().run()
        source_root = Path(__file__).parent / "configs"
        destination = Path(self.build_lib) / "diffusion_lab" / "configs"
        # A previous build must not leave machine-specific configs in the wheel.
        if destination.exists():
            shutil.rmtree(destination)
        for source in sorted(source_root.rglob("*")):
            relative = source.relative_to(source_root)
            if (
                not source.is_file()
                or source.suffix not in {".yaml", ".txt"}
                or "local" in relative.parts
                or any(part.startswith(".") for part in relative.parts)
            ):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


setup(cmdclass={"build_py": BuildWithConfigs})
