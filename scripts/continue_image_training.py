"""Resume a saved local image run and retry bounded transient interruptions.

Reads that run's full config.yaml; the total budget and schedules remain intact.
Slurm users should prefer their site's requeue mechanism and the provided templates.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    if args.max_restarts < 0:
        parser.error("--max-restarts must be nonnegative")
    directory = args.run_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    if args.wait_for_pid:
        process_directory = Path(f"/proc/{args.wait_for_pid}")
        if process_directory.exists():
            command_path = process_directory / "cmdline"
            original = command_path.read_bytes()
            if (
                process_directory.stat().st_uid != os.getuid()
                or b"diffusion_lab.image_train" not in original
                or directory.name.encode() not in original
            ):
                parser.error("Wait PID must belong to this user's image trainer for this run")
            while process_directory.exists():
                try:
                    if (
                        command_path.read_bytes() != original
                        or process_directory.joinpath("stat")
                        .read_text()
                        .split(") ", 1)[1]
                        .split()[0]
                        == "Z"
                    ):
                        break
                except FileNotFoundError:
                    break
                time.sleep(10)
    if not (directory / "config.yaml").exists() or not (directory / "checkpoint.pt").exists():
        parser.error("The run must have both a saved configuration and checkpoint")
    for attempt in range(args.max_restarts + 1):
        if (directory / "training_complete.json").exists():
            print("Configured budget already complete", flush=True)
            return
        command = [
            sys.executable,
            "-m",
            "diffusion_lab.image_train",
            "--config-path",
            str(directory),
            "--config-name",
            "config",
            f"run_dir={directory}",
            f"resume={directory / 'checkpoint.pt'}",
        ]
        print(json.dumps({"resume_attempt": attempt + 1, "run_dir": str(directory)}), flush=True)
        result = subprocess.run(command, cwd=root)
        if result.returncode == 0:
            return
        if attempt == args.max_restarts:
            raise SystemExit(
                f"Training stopped with code {result.returncode}; inspect the log before retrying"
            )
        print(
            json.dumps(
                {"interruption_code": result.returncode, "retry_from_saved_checkpoint": True}
            ),
            flush=True,
        )
        time.sleep(10)


if __name__ == "__main__":
    main()
