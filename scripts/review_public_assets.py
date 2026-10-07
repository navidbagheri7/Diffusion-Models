"""Check public PNG/GIF decoding and measure the source/assets publication footprint."""

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

if __package__:
    from ._publication import publication_files
else:
    from _publication import publication_files


def main():
    root = Path(__file__).resolve().parents[1]
    files = publication_files(root)
    images, frames = 0, 0
    for path in files:
        if path.suffix.lower() not in {".png", ".gif"}:
            continue
        # Decode one immutable byte snapshot, even if a live exporter replaces
        # the public path while this audit is traversing animation frames.
        with Image.open(BytesIO(path.read_bytes())) as image:
            for index in range(getattr(image, "n_frames", 1)):
                image.seek(index)
                image.load()
                frames += 1
        images += 1
    largest = max(files, key=lambda path: path.stat().st_size)
    report = {
        "candidate_public_files": len(files),
        "candidate_source_assets_mib": sum(path.stat().st_size for path in files) / 2**20,
        "largest_candidate_asset": str(largest.relative_to(root)),
        "largest_candidate_asset_mib": largest.stat().st_size / 2**20,
        "verified_images_and_gifs": images,
        "decoded_frames": frames,
        "exported_experiments": len(list((root / "assets/experiments").glob("*/*/metrics.json"))),
        "scope": "Decoding and publication footprint only; quality and scientific correctness have separate reports",
    }
    (root / "assets/reports/diagnostics/public_asset_review.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
