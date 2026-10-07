"""Audit saved image tensors under the current feature/classification protocol.

Keeps the old report and evaluates the saved samples, not regenerated images.
Classifier hashes select the original assessment network. Historical MNIST
reports without a hash use the original noisy digit classifier explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from diffusion_lab.image_guidance import digit_feature_metrics, load_classifier
from diffusion_lab.metrics import image_pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root.resolve()
    candidates = {}
    for path in (root / "outputs").glob("*classifier/classifier.pt"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        candidates[digest] = path
    models = {}
    count = 0
    for path in sorted((root / "outputs").glob("*/metrics.json")):
        report = json.loads(path.read_text())
        if not isinstance(report, dict):
            continue
        is_digit = any(key.startswith("frechet_digit_features_") for key in report)
        if not is_digit and "assessment_classifier_sha256" not in report:
            continue
        if (is_digit and "feature_preprocessing" in report) or (
            not is_digit and "assessment_preprocessing" in report
        ):
            continue
        samples_path = path.parent / "samples.pt"
        if not samples_path.exists():
            continue
        checksum = report.get(
            "feature_classifier_sha256" if is_digit else "assessment_classifier_sha256"
        )
        classifier_path = (
            candidates[checksum] if checksum else root / "outputs/mnist_classifier/classifier.pt"
        )
        if classifier_path not in models:
            models[classifier_path] = load_classifier(classifier_path, args.device)
        model = models[classifier_path]
        tensors = torch.load(samples_path, map_location="cpu", weights_only=False)
        generated, reference = tensors["samples"], tensors["reference"]
        labels = tensors.get("labels")
        if labels is None:
            label = report.get("class_label")
            labels = (
                torch.full((len(generated),), label, dtype=torch.long)
                if label is not None
                else torch.arange(len(generated)) % model.output.out_features
            )
        old = path.parent / "metrics_before_feature_audit.json"
        if not old.exists():
            old.write_text(json.dumps(report, indent=2) + "\n")
        if is_digit:
            report.update(digit_feature_metrics(model, reference, generated, labels, args.device))
        else:
            guesses = []
            with torch.no_grad():
                for batch in generated.split(128):
                    pixels = image_pixels(batch).float().div(127.5).sub(1).to(args.device)
                    time = torch.full((len(batch),), -6.907755278982137 / 4, device=args.device)
                    guesses.append(model(pixels, time).argmax(1).cpu())
            report["requested_class_accuracy"] = (
                (torch.cat(guesses) == labels).float().mean().item()
            )
            report["assessment_preprocessing"] = (
                "Clipped and rounded uint8 images, normalized to [-1,1]"
            )
        report["feature_metric_audit"] = (
            "Recomputed from the original saved image tensors; see metrics_before_feature_audit.json for the previous feature protocol. Generation and Inception scores were not changed."
        )
        path.write_text(json.dumps(report, indent=2) + "\n")
        count += 1
        print(f"Audited {path.parent.name}", flush=True)
    print(f"Audited {count} saved evaluations", flush=True)


if __name__ == "__main__":
    main()
