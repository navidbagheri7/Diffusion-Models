"""Execute a JSON manifest of reproducible experiments, recording failures."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = json.loads(args.manifest.read_text())
    outcomes = []
    for job in jobs:
        output = root / job["output"]
        output.mkdir(parents=True, exist_ok=True)
        marker = output / job.get("marker", "metrics.json")
        if not marker.exists():
            command = [sys.executable, *job["arguments"]]
            if (
                "diffusion_lab.interpolant_experiments" in command
                and "--oracle-only" not in command
                and (output / "checkpoint.pt").exists()
            ):
                command += ["--resume", str(output / "checkpoint.pt")]
            with (output / "process.log").open("a") as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            success = result.returncode == 0 and marker.exists()
        else:
            success = True
        outcomes.append({"output": job["output"], "success": success})
        args.manifest.with_suffix(".results.json").write_text(json.dumps(outcomes, indent=2) + "\n")
        print(json.dumps(outcomes[-1]), flush=True)
    subprocess.run([sys.executable, "scripts/export_assets.py"], cwd=root, check=True)
    if not all(row["success"] for row in outcomes):
        raise SystemExit("Some experiments failed; inspect their process.log files.")


if __name__ == "__main__":
    main()
