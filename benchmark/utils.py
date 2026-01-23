"""Shared utility functions for benchmark modules.

Common helpers that multiple benchmark modules need, like extracting
vocabulary size from different model types.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping
from contextlib import contextmanager
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from torch import nn

from console import logger


def get_model_vocab_size(model: nn.Module, default: int = 32000) -> int:
    """Extract vocabulary size from a model.

    Tries multiple common patterns:
    1. model.vocab_size (simple models, test dummies)
    2. model.config.vocab_size (HuggingFace style)
    3. model.get_input_embeddings().num_embeddings
    4. Falls back to default

    Args:
        model: The model to inspect
        default: Fallback value if vocab size can't be determined

    Returns:
        The vocabulary size as an integer
    """
    # Direct vocab_size attribute
    if hasattr(model, "vocab_size"):
        vocab_size = getattr(model, "vocab_size")
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size
        # Some models expose vocab_size as an optional int property.
        if vocab_size is not None:
            try:
                v = int(vocab_size)
                if v > 0:
                    return v
            except Exception as e:
                logger.error(f"Failed to get vocab size, continuing: {e}")

    # HuggingFace-style config
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):  # type: ignore[union-attr]
        return int(model.config.vocab_size)  # type: ignore[union-attr]

    # Caramba Model: embedder token embedding
    if hasattr(model, "embedder"):
        try:
            emb = getattr(model, "embedder")
            tok_emb = getattr(emb, "token_embedding", None)
            if tok_emb is not None and hasattr(tok_emb, "num_embeddings"):
                return int(tok_emb.num_embeddings)
        except Exception as e:
            logger.error(f"Failed to get vocab size, continuing: {e}")

    # Embedding layer
    if hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()  # type: ignore[operator]
        if embedding is not None and hasattr(embedding, "num_embeddings"):
            return int(embedding.num_embeddings)  # type: ignore[union-attr]

    return default


class LivePlotter:
    """Simple real-time plotter for tracking metrics."""

    def __init__(self, title: str = "Live Plot", *, interactive: bool = True):
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(title)
        self.lines: dict[str, Line2D] = {}
        self.data: dict[str, list[float]] = {}
        self._interactive = bool(interactive)
        
        if self._interactive:
            # Enable interactive mode
            plt.ion()
            plt.show()

    def log(self, **metrics: float) -> None:
        """Log new values for the given metrics."""
        for name, value in metrics.items():
            if name not in self.data:
                self.data[name] = []
                line, = self.ax.plot([], [], label=name)
                self.lines[name] = line
                self.ax.legend()
            
            self.data[name].append(value)
            self.lines[name].set_data(range(len(self.data[name])), self.data[name])
        
        self.ax.relim()
        self.ax.autoscale_view()
        if self._interactive:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def save(self, path: Path) -> None:
        """Save the current figure to disk."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(path, dpi=160, bbox_inches="tight")
        except Exception as e:
            logger.warning(f"Failed to save live plot to {path}: {e!r}")

    def close(self) -> None:
        try:
            if self._interactive:
                plt.ioff()
            plt.close(self.fig)
        except Exception:
            pass


@dataclass
class NullPlotter:
    """No-op plotter (used when realtime plots are disabled)."""

    def log(self, **metrics: float) -> None:  # noqa: ARG002
        return

    def save(self, path: Path) -> None:  # noqa: ARG002
        return

    def close(self) -> None:
        return


@contextmanager
def with_plotter(
    title: str = "Benchmark Metrics", *, enabled: bool = True
) -> Iterator[LivePlotter | NullPlotter]:
    """Context manager for optional live plotting."""
    plotter: LivePlotter | NullPlotter = (
        LivePlotter(title, interactive=True) if enabled else NullPlotter()
    )
    try:
        yield plotter
    finally:
        plotter.close()


def stitch_images(
    *,
    images: list[Path],
    out_path: Path,
    cols: int = 2,
    title: str | None = None,
) -> Path | None:
    """Stitch PNG/JPG images into a single comparison image (grid) using matplotlib."""
    imgs = [p for p in images if p.exists()]
    if not imgs:
        return None

    import math
    import matplotlib.image as mpimg

    n = len(imgs)
    cols = max(1, int(cols))
    rows = int(math.ceil(n / float(cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 4.0 * rows))
    if title:
        fig.suptitle(str(title))

    # Normalize axes to a flat list.
    if isinstance(axes, Axes):
        ax_list = [axes]
    else:
        ax_list = list(axes.flatten())

    for ax in ax_list:
        ax.axis("off")

    for i, p in enumerate(imgs):
        ax = ax_list[i]
        try:
            im = mpimg.imread(str(p))
            ax.imshow(im)
            ax.set_title(p.stem, fontsize=10)
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load\n{p.name}\n{e!r}", ha="center", va="center")
            ax.axis("off")

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        return out_path
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
