"""Static and animated experiment artifacts, with fixed axes across frames."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from torchvision.utils import make_grid

from .metrics import image_pixels


def scatter_comparison(reference, generated, path, title="Generated vs held-out data"):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, values, label in zip(
        axes, [reference, generated], ["Held-out reference", title], strict=True
    ):
        points = values.detach().cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.45)
        ax.set(title=label, xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def particle_animation(trajectory, path, title="Generation", limit=5, labels=None):
    values = trajectory[:, :256].numpy()
    indices = np.unique(np.linspace(0, len(values) - 1, min(40, len(values))).astype(int))
    figure, ax = plt.subplots(figsize=(5, 5))
    coloring = (
        {"c": labels[: values.shape[1]].cpu().numpy(), "cmap": "tab10", "vmin": 0, "vmax": 9}
        if labels is not None
        else {}
    )
    artist = ax.scatter(values[0, :, 0], values[0, :, 1], s=9, alpha=0.6, **coloring)
    ax.set(
        xlim=(-limit, limit), ylim=(-limit, limit), xlabel="$x_1$", ylabel="$x_2$", aspect="equal"
    )

    def update(index):
        artist.set_offsets(values[index])
        ax.set_title(f"{title}: integration step {index}/{len(values) - 1}")
        return (artist,)

    animation = FuncAnimation(figure, update, frames=indices, interval=80)
    animation.save(path, writer=PillowWriter(fps=12))
    plt.close(figure)


def save_rendered_image(images, path, nrow=8):
    """Save precisely the uint8 pixels evaluated by image metrics, including ties."""
    grid = make_grid(image_pixels(images).cpu(), nrow=nrow)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(path)


def image_artifacts(samples, trajectory, directory):
    directory = Path(directory)
    save_rendered_image(samples[:64], directory / "samples.png", nrow=8)
    frames = []
    for index in np.unique(
        np.linspace(0, len(trajectory) - 1, min(24, len(trajectory))).astype(int)
    ):
        frame = make_grid(image_pixels(trajectory[index, :16]).cpu(), nrow=4)
        array = frame.permute(1, 2, 0).numpy()

        frames.append(Image.fromarray(array).resize((384, 384)))
    frames[0].save(
        directory / "generation.gif", save_all=True, append_images=frames[1:], duration=120, loop=0
    )


def loss_plot(history, path):
    figure, ax = plt.subplots(figsize=(7, 4))
    for key in ["train_loss", "validation_loss"]:
        rows = [(entry["step"], entry[key]) for entry in history if key in entry]
        if rows:
            x, y = zip(*rows, strict=True)
            ax.plot(x, y, label=key)
    ax.set(
        xlabel="Optimizer step",
        ylabel="Objective value",
        title="Training and fixed validation draws",
    )
    ax.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
