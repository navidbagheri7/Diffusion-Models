"""Refresh the guidance gallery from saved images and their current metric reports."""

import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "assets/teaching/guidance"
    destination.mkdir(exist_ok=True)
    for dataset in ["mnist", "cifar10"]:
        rows, records = [], []
        for case in ["unconditional", "conditional", "cfg", "cg_low", "cg_high"]:
            folder = root / "outputs" / f"{dataset}_guidance_{case}"
            if not (folder / "metrics.json").exists():
                break
            record = json.loads((folder / "metrics.json").read_text())
            records.append({"case": case, **record})
            with Image.open(folder / "samples.png") as source:
                images = source.convert("RGB").crop((0, 0, source.width, 36))
            row = Image.new("RGB", (images.width, 68), "white")
            ImageDraw.Draw(row).text(
                (4, 2),
                f"{case}: agreement {record.get('requested_class_accuracy', 0):.1%}, NFE {record['nfe_per_image']}",
                fill="black",
            )
            row.paste(images, (0, 30))
            rows.append(row)
        if len(rows) != 5:
            continue
        gallery = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
        offset = 0
        for row in rows:
            gallery.paste(row, (0, offset))
            offset += row.height
        gallery.resize((gallery.width * 2, gallery.height * 2)).save(
            destination / f"{dataset}_comparison.png"
        )
        (destination / f"{dataset}_comparison.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        print(f"Refreshed {dataset} guidance gallery")


if __name__ == "__main__":
    main()
