"""N-model attention visualization for multi-checkpoint comparisons.

Extends the 2-model (teacher/student) attention visualizations to support
arbitrary N models with:
- Shared colorbar [0,1] normalization for fair comparison
- Consistent colors per model
- Flexible layouts for varying number of models and heads

Visualizations:
1. N-row heatmap grid (layer × key position)
2. N-row × H-col last layer heads grid
3. 2N-line attention mass vs depth plot
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False


# Distinct colors for N models
MODEL_COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]


class MultiModelAttentionVisualizer:
    """N-model attention visualization with shared normalization.

    All heatmaps use vmin=0, vmax=1 for fair visual comparison since
    attention weights are probabilities.
    """

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 1.0,
        cmap: str = "viridis",
        head_cmap: str = "magma",
    ):
        """Initialize the visualizer.

        Args:
            vmin: Minimum value for colorbar (default 0.0)
            vmax: Maximum value for colorbar (default 1.0)
            cmap: Colormap for layer-by-token heatmaps
            head_cmap: Colormap for per-head attention matrices
        """
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        self.head_cmap = head_cmap

    def plot_heatmap_comparison(
        self,
        model_events: dict[str, dict[str, Any]],
        tokens: list[str],
        split: int,
        case_id: str,
        output_path: Path,
        title: str | None = None,
    ) -> None:
        """Create N-row heatmap grid with shared colorbar.

        Each row shows a model's layer × key attention heatmap for the
        final query token (averaged over heads).

        Args:
            model_events: Dict mapping model_name -> attention event dict
            tokens: List of token strings for axis labels
            split: Index separating exemplar/target regions (vertical line)
            case_id: Case identifier for title
            output_path: Where to save the figure
            title: Optional custom title
        """
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return

        model_names = list(model_events.keys())
        n_models = len(model_names)

        if n_models == 0:
            return

        # Extract layer-wise attention for each model
        model_data: dict[str, np.ndarray] = {}
        max_layers = 0
        max_tokens = 0

        for model_name, event in model_events.items():
            rows = self._extract_layer_attention(event)
            if rows:
                M = np.stack(rows, axis=0)  # (L, tk)
                model_data[model_name] = M
                max_layers = max(max_layers, M.shape[0])
                max_tokens = max(max_tokens, M.shape[1])

        if not model_data:
            return

        # Create figure with N rows
        fig_height = min(4.0 * n_models, 16.0)
        fig_width = min(14.0, 0.18 * max_tokens + 4.0)
        fig, axes = plt.subplots(n_models, 1, figsize=(fig_width, fig_height))

        # Handle single model case
        if n_models == 1:
            axes = [axes]

        split2 = max(0, min(int(split), max_tokens))

        for i, model_name in enumerate(model_names):
            ax = axes[i]

            if model_name not in model_data:
                ax.text(0.5, 0.5, f"{model_name}: no data",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            M = model_data[model_name]
            L, tk = M.shape

            im = ax.imshow(
                M,
                aspect='auto',
                interpolation='nearest',
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax,
            )
            ax.axvline(split2 - 0.5, color='w', linewidth=1.5, alpha=0.9)
            ax.set_ylabel(model_name, fontsize=10, fontweight='bold')

            # Only show x-axis labels on bottom plot
            if i == n_models - 1:
                ax.set_xlabel("key position (prompt tokens)")
                if tk <= 64:
                    ax.set_xticks(list(range(tk)))
                    ax.set_xticklabels(
                        [t if len(t) <= 6 else t[:6] + "…" for t in tokens[:tk]],
                        rotation=90,
                        fontsize=7
                    )
            else:
                ax.set_xticklabels([])

        # Add shared colorbar on the right side (outside the plot area)
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='attention weight')

        fig_title = title or f"{case_id} • N-Model Attention Heatmaps (final query, mean over heads)"
        fig.suptitle(fig_title, fontsize=12, y=0.98)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def plot_last_layer_heads(
        self,
        model_events: dict[str, dict[str, Any]],
        split: int,
        case_id: str,
        output_path: Path,
        max_heads: int = 4,
        title: str | None = None,
    ) -> None:
        """Create N-row × H-col grid of attention matrices.

        Shows the last layer's per-head attention matrices for all models
        with shared colorbar [0,1].

        Args:
            model_events: Dict mapping model_name -> attention event dict
            split: Index separating exemplar/target regions (vertical line)
            case_id: Case identifier for title
            output_path: Where to save the figure
            max_heads: Maximum number of heads to display per model
            title: Optional custom title
        """
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return

        model_names = list(model_events.keys())
        n_models = len(model_names)

        if n_models == 0:
            return

        # Extract last layer heads for each model
        model_heads: dict[str, list[np.ndarray]] = {}
        max_h = 0

        for model_name, event in model_events.items():
            heads = self._extract_last_layer_heads(event)
            if heads:
                # Limit to max_heads
                heads = heads[:max_heads]
                model_heads[model_name] = heads
                max_h = max(max_h, len(heads))

        if not model_heads or max_h == 0:
            return

        # Create N × H grid
        fig_width = min(20.0, 3.5 * max_h)
        fig_height = min(3.0 * n_models, 12.0)
        fig, axes = plt.subplots(n_models, max_h, figsize=(fig_width, fig_height))

        # Handle edge cases for axes shape
        if n_models == 1 and max_h == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = axes.reshape(1, -1)
        elif max_h == 1:
            axes = axes.reshape(-1, 1)

        split2 = max(0, int(split))

        for i, model_name in enumerate(model_names):
            heads = model_heads.get(model_name, [])

            for j in range(max_h):
                ax = axes[i, j]

                if j < len(heads):
                    a = heads[j]
                    # Ensure split doesn't exceed matrix size
                    s2 = min(split2, a.shape[1] - 1) if a.shape[1] > 0 else 0

                    im = ax.imshow(
                        a,
                        aspect='auto',
                        interpolation='nearest',
                        cmap=self.head_cmap,
                        vmin=self.vmin,
                        vmax=self.vmax,
                    )
                    ax.axvline(s2 - 0.5, color='w', linewidth=0.8, alpha=0.9)

                    # Row label (model name) on leftmost column
                    if j == 0:
                        ax.set_ylabel(model_name, fontsize=9, fontweight='bold')

                    # Column label (head number) on top row
                    if i == 0:
                        ax.set_title(f"head {j}", fontsize=8)
                else:
                    ax.axis('off')

                ax.set_xticks([])
                ax.set_yticks([])

        # Add shared colorbar on the right side (outside the plot area)
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='attention weight')

        fig_title = title or f"{case_id} • N-Model Last Layer Heads (normalized to [0,1])"
        fig.suptitle(fig_title, y=0.98, fontsize=12)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def plot_attention_mass(
        self,
        model_events: dict[str, dict[str, Any]],
        tokens: list[str],
        split: int,
        case_id: str,
        output_path: Path,
        title: str | None = None,
    ) -> None:
        """Create 2N-line plot of attention mass vs depth.

        For each model, plots:
        - Solid line: attention mass on exemplar region (tokens before split)
        - Dashed line: attention mass on target region (tokens after split)

        Args:
            model_events: Dict mapping model_name -> attention event dict
            tokens: List of token strings (for split context)
            split: Index separating exemplar/target regions
            case_id: Case identifier for title
            output_path: Where to save the figure
            title: Optional custom title
        """
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return

        model_names = list(model_events.keys())
        n_models = len(model_names)

        if n_models == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        for i, model_name in enumerate(model_names):
            event = model_events[model_name]
            rows = self._extract_layer_attention(event)

            if not rows:
                continue

            M = np.stack(rows, axis=0)  # (L, tk)
            L, tk = M.shape
            split2 = max(0, min(int(split), tk))

            # Compute attention mass on each region
            mass_exemplar = M[:, :split2].sum(axis=1)
            mass_target = M[:, split2:].sum(axis=1)

            color = MODEL_COLORS[i % len(MODEL_COLORS)]

            ax.plot(
                mass_exemplar,
                label=f"{model_name}: exemplar region",
                linewidth=2,
                linestyle='-',
                color=color,
            )
            ax.plot(
                mass_target,
                label=f"{model_name}: target region",
                linewidth=2,
                linestyle='--',
                color=color,
            )

        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("layer (sampled attention modules)")
        ax.set_ylabel("attention mass (final query token)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=9, ncol=2)

        fig_title = title or f"{case_id} • N-Model Attention Mass vs Depth"
        ax.set_title(fig_title, fontsize=12)
        fig.set_constrained_layout(True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _extract_layer_attention(self, event: dict[str, Any]) -> list[np.ndarray]:
        """Extract per-layer last-row attention (mean over heads).

        Args:
            event: Attention event dict with 'layers' key

        Returns:
            List of 1D arrays, one per layer, containing the mean attention
            distribution for the final query token.
        """
        layers = event.get("layers", [])
        if not isinstance(layers, list) or not layers:
            return []

        rows: list[np.ndarray] = []

        for layer in layers:
            attn = layer.get("attn", None)
            if not isinstance(attn, dict):
                continue
            mats = attn.get("matrices", None)
            if not isinstance(mats, list) or not mats:
                continue

            try:
                head_arrays: list[np.ndarray] = []
                for h in mats:
                    a = np.asarray(h, dtype=np.float32)
                    if a.ndim != 2:
                        continue
                    head_arrays.append(a)
                if not head_arrays:
                    continue

                # Stack heads and get mean of last row
                m = np.stack(head_arrays, axis=0)  # (H, tq, tk)
                last = m[:, -1, :].mean(axis=0)  # (tk,)
                rows.append(last)
            except Exception:
                continue

        return rows

    def _extract_last_layer_heads(self, event: dict[str, Any]) -> list[np.ndarray]:
        """Extract last layer's per-head attention matrices.

        Args:
            event: Attention event dict with 'layers' key

        Returns:
            List of 2D arrays, one per head, containing attention matrices.
        """
        layers = event.get("layers", [])
        if not isinstance(layers, list) or not layers:
            return []

        # Iterate from last layer backwards to find one with valid attention
        for layer in reversed(layers):
            attn = layer.get("attn", None)
            if not isinstance(attn, dict):
                continue
            mats = attn.get("matrices", None)
            if not isinstance(mats, list) or not mats:
                continue

            head_arrays: list[np.ndarray] = []
            for h in mats:
                try:
                    a = np.asarray(h, dtype=np.float32)
                    if a.ndim == 2:
                        head_arrays.append(a)
                except Exception:
                    continue

            if head_arrays:
                return head_arrays

        return []


def render_multi_model_attention_comparison(
    model_events: dict[str, dict[str, Any]],
    tokens: list[str],
    split: int,
    case_id: str,
    output_dir: Path,
    max_heads: int = 4,
) -> dict[str, Path]:
    """Render all N-model attention comparison plots.

    Convenience function that creates all three visualization types:
    1. Layer × key heatmap comparison
    2. Last layer heads comparison
    3. Attention mass vs depth comparison

    Args:
        model_events: Dict mapping model_name -> attention event dict
        tokens: List of token strings
        split: Index separating exemplar/target regions
        case_id: Case identifier for titles
        output_dir: Directory to save figures
        max_heads: Maximum heads to show in head comparison

    Returns:
        Dict mapping artifact names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = MultiModelAttentionVisualizer()
    artifacts: dict[str, Path] = {}

    # 1. Heatmap comparison
    try:
        heatmap_path = output_dir / "attention_heatmap_comparison.png"
        viz.plot_heatmap_comparison(
            model_events=model_events,
            tokens=tokens,
            split=split,
            case_id=case_id,
            output_path=heatmap_path,
        )
        if heatmap_path.exists():
            artifacts["attention_heatmap_comparison"] = heatmap_path
    except Exception:
        pass

    # 2. Last layer heads comparison
    try:
        heads_path = output_dir / "attention_last_layer_heads.png"
        viz.plot_last_layer_heads(
            model_events=model_events,
            split=split,
            case_id=case_id,
            output_path=heads_path,
            max_heads=max_heads,
        )
        if heads_path.exists():
            artifacts["attention_last_layer_heads"] = heads_path
    except Exception:
        pass

    # 3. Attention mass comparison
    try:
        mass_path = output_dir / "attention_mass_vs_depth.png"
        viz.plot_attention_mass(
            model_events=model_events,
            tokens=tokens,
            split=split,
            case_id=case_id,
            output_path=mass_path,
        )
        if mass_path.exists():
            artifacts["attention_mass_vs_depth"] = mass_path
    except Exception:
        pass

    return artifacts
