"""Distribution precision/recall, density/coverage and nearest-neighbor diagnostics.

Neighborhood metrics depend on feature representation, k and sample counts.
They complement FID; they are not calibrated probabilities of visual quality.
Distances are evaluated in blocks to avoid an N x N x feature_dimension tensor.
"""

import argparse
import json
from pathlib import Path

import torch

from .metrics import image_pixels
from .utils import save_json, seed_everything


def neighbor_radii(features, k=3, block_size=256):
    if features.ndim != 2 or not 1 <= k < len(features) or block_size < 1:
        raise ValueError("Neighborhood metrics need a feature matrix, N>k and positive blocks")
    if not torch.isfinite(features).all():
        raise ValueError("Features must be finite")
    values = []
    for offset in range(0, len(features), block_size):
        block = features[offset : offset + block_size]
        distance = torch.cdist(block.double(), features.double())
        distance[
            torch.arange(len(block), device=features.device),
            offset + torch.arange(len(block), device=features.device),
        ] = float("inf")
        values.append(distance.kthvalue(k, dim=1).values)
    return torch.cat(values)


def manifold_metrics(real, generated, k=3, block_size=256):
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("Real/generated feature dimensions must agree")
    real_radius = neighbor_radii(real, k, block_size)
    generated_radius = neighbor_radii(generated, k, block_size)
    precision_hits, density_counts, nearest = [], [], []
    recall_hits = torch.zeros(len(real), device=real.device, dtype=torch.bool)
    coverage_hits = torch.zeros_like(recall_hits)
    for offset in range(0, len(generated), block_size):
        block = generated[offset : offset + block_size]
        distance = torch.cdist(block.double(), real.double())
        inside_real = distance <= real_radius[None]
        precision_hits.append(inside_real.any(1))
        density_counts.append(inside_real.sum(1))
        nearest.append(distance.min(1).values)
        recall_hits |= (distance <= generated_radius[offset : offset + len(block), None]).any(0)
        coverage_hits |= inside_real.any(0)
    nearest = torch.cat(nearest)
    return {
        "precision": torch.cat(precision_hits).float().mean().item(),
        "recall": recall_hits.float().mean().item(),
        "density": (torch.cat(density_counts).double() / k).mean().item(),
        "coverage": coverage_hits.float().mean().item(),
        "nearest_reference_distance_mean": nearest.mean().item(),
        "nearest_reference_distance_median": nearest.median().item(),
        "real_count": len(real),
        "generated_count": len(generated),
        "neighbors_k": k,
        "feature_dimension": real.shape[1],
        "distance": "Euclidean; float64 distance computation; self excluded from kNN radius",
    }


@torch.no_grad()
def inception_features(images, device="cpu", batch_size=64):
    from torchmetrics.image.fid import FrechetInceptionDistance

    extractor = FrechetInceptionDistance(feature=2048, normalize=False).inception.to(device).eval()
    output = [extractor(image_pixels(batch).to(device)).cpu() for batch in images.split(batch_size)]
    return torch.cat(output).float()


def mean_interval(values, confidence=0.95):
    """Student-t interval across genuinely independent repetitions supplied by caller."""
    from scipy.stats import t

    data = torch.tensor(values, dtype=torch.float64)
    if len(data) < 2:
        return {"mean": float(data.mean()), "repetitions": len(data), "confidence_interval": None}
    half = float(
        t.ppf((1 + confidence) / 2, len(data) - 1) * data.std(unbiased=True) / (len(data) ** 0.5)
    )
    return {
        "mean": float(data.mean()),
        "std": float(data.std(unbiased=True)),
        "repetitions": len(data),
        "confidence": confidence,
        "confidence_interval": [float(data.mean()) - half, float(data.mean()) + half],
    }


@torch.no_grad()
def feature_embedding(images, device, classifier=None):
    if images.ndim == 2:
        return images.float()
    if images.shape[1] != 1:
        return inception_features(images, device)
    if classifier is None:
        raise ValueError("Digit metrics require an independent feature classifier")
    output = []
    for batch in images.split(64):
        pixels = image_pixels(batch).to(device).float() / 127.5 - 1
        time = torch.full((len(batch),), -6.907755278982137 / 4, device=device)
        output.append(classifier.features(pixels, time).cpu())
    return torch.cat(output)


def nearest_neighbors(queries, reference, block_size=256):
    """Exact feature nearest neighbors; no random subsampling inside the search."""
    distances, indices = [], []
    for batch in queries.split(block_size):
        values, neighbors = torch.cdist(batch.double(), reference.double()).min(1)
        distances.append(values.cpu())
        indices.append(neighbors.cpu())
    return torch.cat(distances), torch.cat(indices)


def pixel_dataset_config(directory):
    from omegaconf import OmegaConf

    config = OmegaConf.load(directory / "config.yaml").data
    if config.kind == "latents":
        config = OmegaConf.create(
            json.loads((Path(config.root) / "manifest.json").read_text())["image_data"]
        )
    config.horizontal_flip = False
    config.train_limit = config.validation_limit = 0
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--feature-classifier", type=Path)
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--nearest-training", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("assets/reports/distribution_quality"))
    args = parser.parse_args()
    seed_everything(123, 4)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in args.experiment:
        saved = torch.load(directory / "samples.pt", map_location="cpu", weights_only=True)
        generated, reference = saved["samples"], saved["reference"]
        if args.limit <= args.k:
            raise ValueError("Feature limit must exceed k")
        generator = torch.Generator().manual_seed(123)
        gi = torch.randperm(len(generated), generator=generator)[: args.limit]
        ri = torch.randperm(len(reference), generator=generator)[: args.limit]
        generated, reference = generated[gi], reference[ri]
        classifier = None
        if generated.ndim == 2:
            gf, rf = generated, reference
            feature_model = "Original point coordinates"
            feature_hash = None
        elif generated.shape[1] == 1:
            if not args.feature_classifier:
                raise ValueError("Digit metrics require an independent digit feature classifier")
            from .image_guidance import load_classifier

            classifier = load_classifier(args.feature_classifier, args.device)
            gf, rf = (
                feature_embedding(images, args.device, classifier)
                for images in [generated, reference]
            )
            feature_model = "Independent digit classifier, sigma=.001; rendered-pixel preprocessing"
            feature_hash = classifier.checkpoint_sha256
        else:
            gf, rf = (
                inception_features(generated, args.device),
                inception_features(reference, args.device),
            )
            feature_model = "Torch-fidelity Inception pool3, 2048 dimensions; rendered uint8 images"
            feature_hash = None
        metrics = manifold_metrics(rf.to(args.device), gf.to(args.device), args.k, args.block_size)
        protocol = (
            json.loads((directory / "metrics.json").read_text())
            if (directory / "metrics.json").exists()
            else {}
        )
        metrics.update(
            {
                "experiment": directory.name,
                "feature_model": feature_model,
                "feature_classifier_sha256": feature_hash,
                "subsample_seed": 123,
                "generation_seed": protocol.get("seed"),
                "checkpoint_sha256": protocol.get("checkpoint_sha256"),
                "counts_are_subsamples_of_saved_outputs": True,
                "precision_recall_reference": "Kynkaanniemi et al. 2019",
                "density_coverage_reference": "Naeem et al. 2020",
            }
        )
        folder = args.output / directory.name
        folder.mkdir(exist_ok=True)
        if args.per_class and saved.get("labels") is not None:
            labels = saved["labels"][gi]
            reference_labels = saved.get("reference_labels")
            if reference_labels is None and reference.ndim == 4:
                from .data import ImageDataset, balanced_image_indices

                dataset = ImageDataset(
                    pixel_dataset_config(directory), protocol.get("reference_split", "validation")
                )
                selected = balanced_image_indices(
                    dataset, len(saved["reference"]), protocol.get("class_label")
                )
                reference_labels = torch.tensor([dataset.files[i][1] for i in selected])
            if reference_labels is not None:
                reference_labels = reference_labels[ri]
                by_class = {}
                for label in labels.unique().tolist():
                    real_subset, generated_subset = (
                        rf[reference_labels == label],
                        gf[labels == label],
                    )
                    if min(len(real_subset), len(generated_subset)) > args.k:
                        by_class[str(label)] = manifold_metrics(
                            real_subset.to(args.device),
                            generated_subset.to(args.device),
                            args.k,
                            args.block_size,
                        )
                metrics["per_requested_class"] = by_class
                metrics["class_interpretation"] = (
                    "Generation labels are requested classes, not independently inferred classes"
                )
        if args.nearest_training and generated.ndim == 4:
            from .data import ImageDataset
            from .visualize import save_rendered_image

            dataset = ImageDataset(pixel_dataset_config(directory), "train")
            training_indices = torch.randperm(len(dataset), generator=generator)[: len(reference)]
            training_images = torch.stack([dataset[int(i)][0] for i in training_indices])
            training_features = feature_embedding(training_images, args.device, classifier)
            td, ti = nearest_neighbors(
                gf.to(args.device), training_features.to(args.device), args.block_size
            )
            vd, vi = nearest_neighbors(gf.to(args.device), rf.to(args.device), args.block_size)
            metrics["training_neighbor_diagnostic"] = {
                "training_count": len(training_features),
                "heldout_count": len(rf),
                "training_distance_mean": float(td.mean()),
                "heldout_distance_mean": float(vd.mean()),
                "mean_training_to_heldout_distance_ratio": float(
                    td.mean() / vd.mean().clamp_min(1e-12)
                ),
                "scope": "Equal-size random training/held-out feature neighbors; not proof of non-memorization",
            }
            gallery = torch.stack(
                [generated[:16], training_images[ti[:16]], reference[vi[:16]]], 1
            ).flatten(0, 1)
            save_rendered_image(gallery, folder / "nearest_neighbors.png", nrow=3)
            save_json(
                folder / "nearest_neighbors.json",
                {
                    "columns": ["generated", "training neighbor", "held-out neighbor"],
                    "training_dataset_indices": training_indices[ti].tolist(),
                    "heldout_saved_indices": ri[vi].tolist(),
                    "training_distances": td.tolist(),
                    "heldout_distances": vd.tolist(),
                },
            )
        save_json(folder / "metrics.json", metrics)
        torch.save(
            {"real": rf, "generated": gf, "real_indices": ri, "generated_indices": gi},
            directory / "distribution_features.pt",
        )
        rows.append(metrics)
        print(json.dumps(metrics), flush=True)
    save_json(args.output / "comparison.json", rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in rows:
        label = row["experiment"]
        axes[0].scatter(row["recall"], row["precision"], label=label)
        axes[1].scatter(row["coverage"], row["density"], label=label)
    axes[0].set(
        xlabel="Recall: real manifold covered",
        ylabel="Precision: generated samples inside real manifold",
        xlim=(0, 1.05),
        ylim=(0, 1.05),
    )
    axes[1].set(xlabel="Coverage", ylabel="Density", xlim=(0, 1.05))
    axes[1].legend(fontsize=7)
    fig.suptitle("Complementary distribution diagnostics; compare only matching datasets/protocols")
    fig.tight_layout()
    fig.savefig(args.output / "precision_recall.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
