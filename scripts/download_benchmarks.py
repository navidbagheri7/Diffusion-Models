"""Download official torchvision benchmarks and verify every exported image.

Raw archives retain torchvision's published checksum verification. Images are
exported without resizing into the same labeled layout as the other datasets.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST


def export(root, name, dataset_class):
    destination = root / name
    class_names = None
    counts = {}
    for train, split in [(True, "train"), (False, "validation")]:
        data = dataset_class(str(destination / "raw"), train=train, download=True)
        class_names = list(data.classes)
        counts[split] = [0] * 10
        digest = hashlib.sha256()
        for index in range(len(data)):
            image, label = data[index]
            folder = destination / "images" / split / f"class_{label:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{index:06d}.png"
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                image.save(temporary, format="PNG")
                temporary.replace(path)
            with Image.open(path) as check:
                check.load()
                assert check.size == image.size and check.mode == image.mode
                assert np.array_equal(np.asarray(check), np.asarray(image)), path
            digest.update(path.read_bytes())
            counts[split][label] += 1
        print(
            json.dumps({"dataset": name, "split": split, "count": len(data), "verified": True}),
            flush=True,
        )
        counts[f"{split}_sha256_in_index_order"] = digest.hexdigest()
    report = {
        "dataset": name,
        "source": "torchvision official dataset with archive checksums",
        "classes": class_names,
        "counts": counts,
        "all_pixels_verified": True,
        "official_test_used_as_validation": True,
    }
    (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tii/Image_data"))
    args = parser.parse_args()
    for name, cls in [("cifar10", CIFAR10), ("mnist", MNIST)]:
        export(args.root, name, cls)


if __name__ == "__main__":
    main()
