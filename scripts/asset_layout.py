"""Shared dataset grouping for public experiment exporters and galleries."""


def experiment_group(name: str) -> str:
    """Choose a stable browsing category; this does not assess run quality."""
    if name.startswith("likelihood_"):
        return "likelihood"
    if name.startswith(("dsbm_", "gmm_bridge_")):
        return "bridges"
    for prefix, group in (
        ("cifar10_", "cifar10"),
        ("mnist_", "mnist"),
        ("celeba", "faces"),
        ("ffhq_", "faces"),
        ("coco_text", "text"),
        ("imagenet_", "imagenet"),
    ):
        if name.startswith(prefix):
            return group
    if name.startswith(
        (
            "gmm_",
            "cfg_",
            "classifier_",
            "checkerboard_",
            "olympic_rings_",
            "wavy_rings_",
            "two_spirals_",
            "interpolant_",
        )
    ):
        return "toy"
    return "custom"
