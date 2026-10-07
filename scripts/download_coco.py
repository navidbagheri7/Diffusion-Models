"""Download official COCO 2017 images/captions, verify, and build portable manifests.

Only caption annotations are extracted. All image ZIP members are CRC-checked
and all extracted JPEGs are decoded. HTTPS uses the official bucket's path-style
endpoint, avoiding the certificate mismatch on images.cocodataset.org.
"""

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from diffusion_lab.utils import save_json

BASE = "https://s3.amazonaws.com/images.cocodataset.org"
ARCHIVES = {
    "train2017.zip": f"{BASE}/zips/train2017.zip",
    "val2017.zip": f"{BASE}/zips/val2017.zip",
    "annotations_trainval2017.zip": f"{BASE}/annotations/annotations_trainval2017.zip",
}


def download(url, destination, connections=1):
    destination.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, timeout=60)
    head.raise_for_status()
    size, etag = int(head.headers["Content-Length"]), head.headers.get("ETag")
    metadata = destination.with_suffix(".remote.json")
    partial = destination.with_suffix(".part")
    identity = {"url": url, "bytes": size, "etag": etag}
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError(f"Existing archive has unexpected size: {destination}")
    else:
        if partial.exists() and (
            not metadata.exists() or json.loads(metadata.read_text()) != identity
        ):
            raise ValueError(f"Remote identity changed for partial download: {partial}")
        save_json(metadata, identity)
        if connections > 1:
            chunk_size = 32 * 1024 * 1024
            folder = destination.parent / (destination.stem + "_chunks")
            folder.mkdir(exist_ok=True)
            # Preserve an earlier sequential download by splitting its prefix.
            if partial.exists() and not any(folder.iterdir()):
                with partial.open("rb") as prefix:
                    for index in range((partial.stat().st_size + chunk_size - 1) // chunk_size):
                        (folder / f"{index:05d}.part").write_bytes(prefix.read(chunk_size))

            def get_chunk(index):
                start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
                part = folder / f"{index:05d}.part"
                for attempt in range(12):
                    have = part.stat().st_size if part.exists() else 0
                    if have == end - start + 1:
                        return
                    if have > end - start + 1:
                        raise ValueError("Oversized range chunk")
                    try:
                        with requests.get(
                            url,
                            headers={"Range": f"bytes={start + have}-{end}"},
                            stream=True,
                            timeout=(30, 90),
                        ) as response:
                            response.raise_for_status()
                            expected = f"bytes {start + have}-{end}/{size}"
                            if (
                                response.status_code != 206
                                or response.headers.get("Content-Range") != expected
                            ):
                                raise ValueError("Incorrect HTTP range response")
                            with part.open("ab") as handle:
                                for content in response.iter_content(1024 * 1024):
                                    handle.write(content)
                        if part.stat().st_size != end - start + 1:
                            raise OSError("Range transfer incomplete")
                        return
                    except (requests.RequestException, OSError):
                        if attempt == 11:
                            raise
                        time.sleep(min(10, attempt + 1))
                raise OSError("Range retry budget exhausted")

            count = (size + chunk_size - 1) // chunk_size
            with ThreadPoolExecutor(max_workers=connections) as pool:
                for completed, _ in enumerate(pool.map(get_chunk, range(count)), 1):
                    if completed % 16 == 0 or completed == count:
                        print(
                            f"{destination.name}: {completed}/{count} ranges complete", flush=True
                        )
            with partial.open("wb") as assembled:
                for index in range(count):
                    with (folder / f"{index:05d}.part").open("rb") as source:
                        shutil.copyfileobj(source, assembled, 1024 * 1024)
            if partial.stat().st_size != size:
                raise OSError("Assembled archive size mismatch")
            partial.replace(destination)
            shutil.rmtree(folder)
            return download(url, destination, 1)
        for attempt in range(12):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == size:
                break
            if offset > size:
                raise ValueError("Partial archive is larger than remote object")
            try:
                with requests.get(
                    url,
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                    stream=True,
                    timeout=(30, 90),
                ) as response:
                    response.raise_for_status()
                    if offset and (
                        response.status_code != 206
                        or not response.headers.get("Content-Range", "").startswith(
                            f"bytes {offset}-"
                        )
                    ):
                        raise ValueError("Server did not honor the resume offset")
                    last_report = offset
                    with partial.open("ab" if offset else "wb") as handle:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            handle.write(chunk)
                            offset += len(chunk)
                            if offset - last_report >= 256 * 1024 * 1024:
                                print(f"{destination.name}: {offset / size:.1%}", flush=True)
                                last_report = offset
                if offset != size:
                    raise OSError("Archive transfer ended at the wrong size")
                break
            except (requests.RequestException, OSError) as error:
                if attempt == 11:
                    raise
                print(f"Retry {destination.name}: {error}", flush=True)
                time.sleep(min(10, 1 + attempt))
        if not partial.exists() or partial.stat().st_size != size:
            raise OSError("Archive download incomplete")
        partial.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {**identity, "sha256": digest.hexdigest(), "archive": destination.name}


def extract(archive, root):
    with zipfile.ZipFile(archive) as compressed:
        for member in compressed.infolist():
            name = Path(member.filename)
            if member.is_dir():
                continue
            if "annotations" in name.parts and name.name not in {
                "captions_train2017.json",
                "captions_val2017.json",
            }:
                continue
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Archive member escapes dataset directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".extracting")
            # Reading each member to EOF verifies its ZIP CRC, including on reruns.
            with compressed.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            temporary.replace(target)
    print(f"Extracted and CRC-checked {archive.name}", flush=True)


def verify_image(item):
    path, expected_width, expected_height = item
    with Image.open(path) as image:
        image.load()
        if image.size != (expected_width, expected_height):
            raise ValueError(f"Image size disagrees with annotation: {path}")
        image.convert("RGB")


def build_manifest(root, split, workers):
    annotation = root / "annotations" / f"captions_{split}2017.json"
    data = json.loads(annotation.read_text())
    captions = defaultdict(list)
    for item in data["annotations"]:
        caption = item["caption"].strip()
        if not caption:
            raise ValueError("Empty COCO caption")
        captions[item["image_id"]].append(caption)
    images = sorted(data["images"], key=lambda item: item["id"])
    if set(captions) != {image["id"] for image in images}:
        raise ValueError("Caption and image IDs differ")
    expected = 118287 if split == "train" else 5000
    if len(images) != expected:
        raise ValueError(f"Unexpected COCO {split} count")
    checks = [
        (root / f"{split}2017" / item["file_name"], item["width"], item["height"])
        for item in images
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(pool.map(verify_image, checks), 1):
            if count % 10000 == 0:
                print(f"Decoded {count}/{len(images)} {split} images", flush=True)
    destination = root / f"{split}.jsonl"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w") as output:
        for image in images:
            output.write(
                json.dumps(
                    {
                        "image_id": image["id"],
                        "image": f"{split}2017/{image['file_name']}",
                        "captions": captions[image["id"]],
                        "width": image["width"],
                        "height": image["height"],
                    }
                )
                + "\n"
            )
    temporary.replace(destination)
    return {
        "images": len(images),
        "captions": len(data["annotations"]),
        "all_images_decoded": True,
        "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tii/Image_data/coco2017"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--connections", type=int, default=16)
    args = parser.parse_args()
    if min(args.workers, args.connections) < 1:
        raise ValueError("workers must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    if (
        shutil.disk_usage(args.root).free < 50 * 1024**3
        and not (args.root / "verification.json").exists()
    ):
        raise OSError("Reserve at least 50 GiB for COCO archives and extracted images")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(download, url, args.root / "archives" / name, args.connections)
            for name, url in ARCHIVES.items()
        ]
        archives = [future.result() for future in futures]
    for archive in archives:
        extract(args.root / "archives" / archive["archive"], args.root)
    splits = {split: build_manifest(args.root, split, args.workers) for split in ["train", "val"]}
    save_json(
        args.root / "verification.json",
        {
            "dataset": "COCO 2017 captions",
            "source": "https://cocodataset.org/#download",
            "archives": archives,
            "splits": splits,
            "checksum_scope": "Locally recorded SHA-256; ZIP CRC and full JPEG decoding. No upstream SHA-256 claim.",
            "terms": "https://cocodataset.org/#termsofuse",
        },
    )
    print(json.dumps(splits), flush=True)


if __name__ == "__main__":
    main()
