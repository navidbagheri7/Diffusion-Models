"""Explicit finite-sample metrics; no pixel-space quantity is called FID."""

import torch
from scipy.optimize import linear_sum_assignment


def image_pixels(images):
    """Evaluate rendered images using identical clipping and 8-bit quantization."""
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def mmd_squared(x, y, bandwidths=(0.25, 0.5, 1.0, 2.0), block_size=256):
    """Unbiased squared MMD with an average of RBF kernels; may be negative."""
    if min(len(x), len(y)) < 2:
        raise ValueError("Unbiased MMD requires at least two samples per set")

    def kernel(a, b):
        distances = torch.cdist(a, b).square()
        return torch.stack([torch.exp(-distances / (2 * h**2)) for h in bandwidths]).mean(0)

    def total(a, b, diagonal=False):
        value = torch.zeros((), device=a.device, dtype=torch.float64)
        for row in range(0, len(a), block_size):
            for col in range(0, len(b), block_size):
                block = kernel(a[row : row + block_size], b[col : col + block_size])
                value = value + block.double().sum()
                if diagonal and row == col:
                    value = value - block.diag().double().sum()
        return value

    return (
        total(x, x, True) / (len(x) * (len(x) - 1))
        + total(y, y, True) / (len(y) * (len(y) - 1))
        - 2 * total(x, y) / (len(x) * len(y))
    ).item()


def sliced_wasserstein(x, y, projections=256, seed=0):
    """Sliced W2: sqrt(mean projected squared quantile differences)."""
    if len(x) != len(y):
        raise ValueError("This exact empirical quantile implementation requires equal counts")
    generator = torch.Generator(device=x.device).manual_seed(seed)
    directions = torch.randn(x.shape[1], projections, device=x.device, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True)
    return (
        ((x @ directions).sort(0).values - (y @ directions).sort(0).values)
        .square()
        .mean()
        .sqrt()
        .item()
    )


def empirical_wasserstein(x, y):
    """Exact W2 between equal-weight, equal-size empirical measures, via assignment."""
    if len(x) != len(y):
        raise ValueError("Exact assignment W2 requires equal sample counts")
    cost = torch.cdist(x, y).square().cpu().double().numpy()
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean() ** 0.5)


def gmm_metrics(samples, reference, mixture):
    distances = torch.cdist(samples, mixture.means)
    labels = distances.argmin(1)
    # Define an in-distribution point as lying within 3 component standard deviations.
    accepted = distances.min(1).values <= 3 * mixture.std
    counts = torch.bincount(labels[accepted], minlength=mixture.components).float()
    probabilities = counts / len(samples)
    return {
        "mmd_squared_unbiased": mmd_squared(samples, reference),
        "sliced_w2": sliced_wasserstein(samples, reference),
        "empirical_w2_subsample": empirical_wasserstein(samples[:512], reference[:512]),
        "wasserstein_sample_count": min(512, len(samples)),
        "mean_target_negative_log_density": -mixture.log_prob(samples).mean().item(),
        "mode_coverage_3sigma": (counts > 0).float().mean().item(),
        "fraction_within_3sigma": accepted.float().mean().item(),
        "mode_total_variation_with_outlier_bin": (
            (probabilities - 1 / mixture.components).abs().sum().item()
            + 1
            - accepted.float().mean().item()
        )
        / 2,
    }


def image_metrics(
    real, generated, device="cpu", batch_size=32, splits=5, kid_subset_size=1000, kid_subsets=50
):
    """FID/KID/IS using torch-fidelity's pretrained Inception implementation.

    Inputs: RGB NCHW tensors in [-1,1]. uint8 conversion occurs identically for both
    sets. Covariance FID uses 2048 features, so small runs have very noisy estimates.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.kid import KernelInceptionDistance

    if min(len(real), len(generated)) < 2:
        raise ValueError("Image metrics need at least two examples")
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        subsets=kid_subsets,
        subset_size=min(kid_subset_size, len(real), len(generated)),
        normalize=False,
    ).to(device)
    inception = InceptionScore(splits=min(splits, len(generated)), normalize=False).to(device)
    for images, is_real in [(real, True), (generated, False)]:
        for batch in images.split(batch_size):
            pixels = image_pixels(batch).to(device)
            fid.update(pixels, real=is_real)
            kid.update(pixels, real=is_real)
            if not is_real:
                inception.update(pixels)
    kid_mean, kid_std = kid.compute()
    is_mean, is_std = inception.compute()
    return {
        "fid_2048": fid.compute().item(),
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
        "inception_score_mean": is_mean.item(),
        "inception_score_std": is_std.item(),
        "real_count": len(real),
        "generated_count": len(generated),
        "kid_subsets": kid_subsets,
        "kid_subset_size": min(kid_subset_size, len(real), len(generated)),
        "inception_score_splits": min(splits, len(generated)),
        "preprocessing": "RGB [-1,1] -> rounded uint8; torch-fidelity Inception resize",
        "small_sample_warning": min(len(real), len(generated)) < 10000,
    }
