"""
Attention analysis for adversarial testing.

Provides tools for:
1. Comparing attention between clean and attacked prompts
2. Tracking attention shifts under attack
3. Visualizing attention differences
4. Extracting attention statistics for scoring
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class AttentionStats:
    """Statistics about attention patterns."""
    entropy: float  # Higher = more uniform attention
    sparsity: float  # Fraction of near-zero weights
    peak_concentration: float  # Max attention weight
    diagonal_ratio: float  # How much attention is on diagonal (self)
    head_agreement: float  # How similar are different heads
    layer_variance: float  # Variance across layers

    def to_dict(self) -> dict[str, float]:
        return {
            "entropy": self.entropy,
            "sparsity": self.sparsity,
            "peak_concentration": self.peak_concentration,
            "diagonal_ratio": self.diagonal_ratio,
            "head_agreement": self.head_agreement,
            "layer_variance": self.layer_variance,
        }


@dataclass
class AttentionDiff:
    """Difference between two attention patterns."""
    clean_stats: AttentionStats
    attacked_stats: AttentionStats
    diff_magnitude: float  # Overall difference
    entropy_shift: float  # Change in entropy
    peak_shift: float  # Change in peak concentration
    affected_layers: list[int]  # Layers with significant changes
    affected_heads: list[tuple[int, int]]  # (layer, head) pairs affected

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_stats": self.clean_stats.to_dict(),
            "attacked_stats": self.attacked_stats.to_dict(),
            "diff_magnitude": self.diff_magnitude,
            "entropy_shift": self.entropy_shift,
            "peak_shift": self.peak_shift,
            "affected_layers": self.affected_layers,
            "affected_heads": self.affected_heads,
        }


@dataclass
class TokenAttentionAnalysis:
    """Token-level attention analysis for failure diagnosis."""
    token_ids: list[int]
    tokens: list[str]
    attention_received: list[float]  # How much attention each token receives
    attention_given: list[float]  # How much attention each token gives
    critical_tokens: list[int]  # Indices of tokens receiving abnormal attention
    distractor_tokens: list[int]  # Indices identified as distractors
    target_token_idx: int | None  # Index of the target/answer token


def compute_attention_stats(attention: np.ndarray) -> AttentionStats:
    """
    Compute comprehensive statistics about attention patterns.

    Args:
        attention: Attention tensor [layers, heads, seq, seq]

    Returns:
        AttentionStats object
    """
    eps = 1e-10

    # Entropy (measure of uniformity)
    # Higher entropy = more uniform attention
    entropy_per_position = -np.sum(attention * np.log(attention + eps), axis=-1)
    avg_entropy = float(entropy_per_position.mean())

    # Sparsity (fraction of weights below threshold)
    sparsity = float((attention < 0.01).mean())

    # Peak concentration
    peak_concentration = float(attention.max())

    # Diagonal ratio (self-attention)
    n_layers, n_heads, seq_len, _ = attention.shape
    diagonal_mask = np.eye(seq_len, dtype=bool)
    diagonal_attention = attention[:, :, diagonal_mask].mean()
    diagonal_ratio = float(diagonal_attention)

    # Head agreement (cosine similarity between heads)
    # Flatten each head's attention, compute pairwise similarity
    flat_heads = attention.reshape(n_layers * n_heads, -1)
    norms = np.linalg.norm(flat_heads, axis=1, keepdims=True) + eps
    normalized = flat_heads / norms
    similarity_matrix = normalized @ normalized.T
    # Take mean of upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n_layers * n_heads, k=1)
    head_agreement = float(similarity_matrix[triu_indices].mean())

    # Layer variance
    layer_means = attention.mean(axis=(1, 2, 3))  # Mean per layer
    layer_variance = float(layer_means.var())

    return AttentionStats(
        entropy=avg_entropy,
        sparsity=sparsity,
        peak_concentration=peak_concentration,
        diagonal_ratio=diagonal_ratio,
        head_agreement=head_agreement,
        layer_variance=layer_variance,
    )


def compare_attention(
    clean_attention: np.ndarray,
    attacked_attention: np.ndarray,
    threshold: float = 0.1,
) -> AttentionDiff:
    """
    Compare attention patterns between clean and attacked prompts.

    Args:
        clean_attention: Attention from clean prompt [layers, heads, seq, seq]
        attacked_attention: Attention from attacked prompt
        threshold: Threshold for considering a change significant

    Returns:
        AttentionDiff object with comparison metrics
    """
    clean_stats = compute_attention_stats(clean_attention)
    attacked_stats = compute_attention_stats(attacked_attention)

    # Compute difference magnitude
    # Need to handle potentially different sequence lengths
    min_seq = min(clean_attention.shape[-1], attacked_attention.shape[-1])
    clean_trimmed = clean_attention[:, :, :min_seq, :min_seq]
    attacked_trimmed = attacked_attention[:, :, :min_seq, :min_seq]

    diff = np.abs(attacked_trimmed - clean_trimmed)
    diff_magnitude = float(diff.mean())

    # Entropy shift
    entropy_shift = attacked_stats.entropy - clean_stats.entropy

    # Peak shift
    peak_shift = attacked_stats.peak_concentration - clean_stats.peak_concentration

    # Find affected layers (those with above-threshold mean difference)
    layer_diffs = diff.mean(axis=(1, 2, 3))
    affected_layers = [int(i) for i, d in enumerate(layer_diffs) if d > threshold]

    # Find affected heads
    head_diffs = diff.mean(axis=(2, 3))  # [layers, heads]
    affected_heads = []
    for layer_idx in range(head_diffs.shape[0]):
        for head_idx in range(head_diffs.shape[1]):
            if head_diffs[layer_idx, head_idx] > threshold:
                affected_heads.append((int(layer_idx), int(head_idx)))

    return AttentionDiff(
        clean_stats=clean_stats,
        attacked_stats=attacked_stats,
        diff_magnitude=diff_magnitude,
        entropy_shift=entropy_shift,
        peak_shift=peak_shift,
        affected_layers=affected_layers,
        affected_heads=affected_heads,
    )


def analyze_token_attention(
    attention: np.ndarray,
    tokens: list[str],
    target_token: str | None = None,
    layer_idx: int = -1,  # -1 for last layer
) -> TokenAttentionAnalysis:
    """
    Analyze attention at the token level.

    Args:
        attention: Attention tensor [layers, heads, seq, seq]
        tokens: List of token strings
        target_token: The expected answer token (if known)
        layer_idx: Which layer to analyze (-1 for last)

    Returns:
        TokenAttentionAnalysis with per-token metrics
    """
    n_layers, n_heads, seq_len, _ = attention.shape

    if layer_idx < 0:
        layer_idx = n_layers + layer_idx

    layer_attention = attention[layer_idx]  # [heads, seq, seq]

    # Average across heads
    avg_attention = layer_attention.mean(axis=0)  # [seq, seq]

    # Attention received: sum of attention TO each token (column sum)
    attention_received = avg_attention.sum(axis=0).tolist()

    # Attention given: sum of attention FROM each token (row sum)
    attention_given = avg_attention.sum(axis=1).tolist()

    # Find critical tokens (those receiving unusually high attention)
    mean_received = np.mean(attention_received)
    std_received = np.std(attention_received)
    threshold = mean_received + 2 * std_received
    critical_tokens = [i for i, v in enumerate(attention_received) if v > threshold]

    # Identify distractor tokens (high attention but not the target)
    distractor_tokens = []
    target_token_idx = None

    if target_token:
        # Find target token
        for i, t in enumerate(tokens):
            if target_token.lower() in t.lower():
                target_token_idx = i
                break

        # Distractors are critical tokens that aren't the target
        distractor_tokens = [i for i in critical_tokens if i != target_token_idx]

    return TokenAttentionAnalysis(
        token_ids=list(range(len(tokens))),
        tokens=tokens,
        attention_received=attention_received,
        attention_given=attention_given,
        critical_tokens=critical_tokens,
        distractor_tokens=distractor_tokens,
        target_token_idx=target_token_idx,
    )


def measure_degeneration(outputs: list[str]) -> dict[str, float]:
    """
    Measure repetition/degeneration in model outputs.

    Args:
        outputs: List of generated outputs

    Returns:
        Dictionary of degeneration metrics
    """
    metrics = {}

    for i, output in enumerate(outputs):
        tokens = output.split()

        if not tokens:
            metrics[f"output_{i}_unique_ratio"] = 0.0
            metrics[f"output_{i}_max_repeat"] = 0
            continue

        # Unique token ratio
        unique_ratio = len(set(tokens)) / len(tokens)
        metrics[f"output_{i}_unique_ratio"] = unique_ratio

        # Find maximum consecutive repetition
        max_repeat = 1
        current_repeat = 1
        for j in range(1, len(tokens)):
            if tokens[j] == tokens[j - 1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1

        metrics[f"output_{i}_max_repeat"] = max_repeat

        # Repetition ratio (fraction of tokens that are repeats)
        repeat_count = sum(1 for j in range(1, len(tokens)) if tokens[j] == tokens[j - 1])
        metrics[f"output_{i}_repeat_ratio"] = repeat_count / max(len(tokens) - 1, 1)

    return metrics


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_attention_diff(
    clean_attention: np.ndarray,
    attacked_attention: np.ndarray,
    layer_idx: int = -1,
    head_idx: int = 0,
    clean_tokens: list[str] | None = None,
    attacked_tokens: list[str] | None = None,
    output_path: Path | None = None,
    title: str = "Attention Difference: Clean vs Attacked",
) -> None:
    """
    Create a side-by-side attention heatmap comparison.

    Args:
        clean_attention: Attention from clean prompt
        attacked_attention: Attention from attacked prompt
        layer_idx: Layer to visualize
        head_idx: Head to visualize
        clean_tokens: Token labels for clean prompt
        attacked_tokens: Token labels for attacked prompt
        output_path: Where to save the figure
        title: Figure title
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available for visualization")
        return

    n_layers = clean_attention.shape[0]
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx

    clean_layer = clean_attention[layer_idx, head_idx]
    attacked_layer = attacked_attention[layer_idx, head_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Common colorbar range
    vmin = min(clean_layer.min(), attacked_layer.min())
    vmax = max(clean_layer.max(), attacked_layer.max())

    # Clean attention
    im1 = axes[0].imshow(clean_layer, cmap='Blues', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title("Clean Prompt")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    if clean_tokens:
        n = min(len(clean_tokens), clean_layer.shape[0])
        axes[0].set_xticks(range(n))
        axes[0].set_xticklabels(clean_tokens[:n], rotation=45, ha='right', fontsize=7)
        axes[0].set_yticks(range(n))
        axes[0].set_yticklabels(clean_tokens[:n], fontsize=7)

    # Attacked attention
    im2 = axes[1].imshow(attacked_layer, cmap='Blues', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title("Attacked Prompt")
    axes[1].set_xlabel("Key Position")
    if attacked_tokens:
        n = min(len(attacked_tokens), attacked_layer.shape[0])
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(attacked_tokens[:n], rotation=45, ha='right', fontsize=7)
        axes[1].set_yticks(range(n))
        axes[1].set_yticklabels(attacked_tokens[:n], fontsize=7)

    # Difference (need to align sizes)
    min_seq = min(clean_layer.shape[0], attacked_layer.shape[0])
    diff = attacked_layer[:min_seq, :min_seq] - clean_layer[:min_seq, :min_seq]

    # Diverging colormap for difference
    abs_max = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-abs_max, vmax=abs_max)
    axes[2].set_title("Difference (Attacked - Clean)")
    axes[2].set_xlabel("Key Position")
    plt.colorbar(im3, ax=axes[2], label="Δ Attention")

    fig.suptitle(f"{title}\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_layer_summary(
    attention_diff: AttentionDiff,
    clean_attention: np.ndarray,
    attacked_attention: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """
    Create a summary visualization of attention changes across layers.

    Args:
        attention_diff: AttentionDiff object
        clean_attention: Full clean attention tensor
        attacked_attention: Full attacked attention tensor
        output_path: Where to save
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available for visualization")
        return

    n_layers = clean_attention.shape[0]

    # Compute per-layer metrics
    min_seq = min(clean_attention.shape[-1], attacked_attention.shape[-1])
    clean_trimmed = clean_attention[:, :, :min_seq, :min_seq]
    attacked_trimmed = attacked_attention[:, :, :min_seq, :min_seq]

    layer_diff_means = np.abs(attacked_trimmed - clean_trimmed).mean(axis=(1, 2, 3))

    eps = 1e-10
    clean_entropy = -np.sum(clean_trimmed * np.log(clean_trimmed + eps), axis=-1).mean(axis=(1, 2))
    attacked_entropy = -np.sum(attacked_trimmed * np.log(attacked_trimmed + eps), axis=-1).mean(axis=(1, 2))
    entropy_diff = attacked_entropy - clean_entropy

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Layer difference magnitude
    axes[0, 0].bar(range(n_layers), layer_diff_means)
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Mean Absolute Difference")
    axes[0, 0].set_title("Attention Change by Layer")
    for i in attention_diff.affected_layers:
        if i < n_layers:
            axes[0, 0].axvline(i, color='red', alpha=0.3, linestyle='--')

    # Entropy shift
    axes[0, 1].bar(range(n_layers), entropy_diff)
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Entropy Shift")
    axes[0, 1].set_title("Entropy Change by Layer (+ = more uniform)")
    axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Stats comparison
    stats_names = ['Entropy', 'Sparsity', 'Peak', 'Diag Ratio', 'Head Agree']
    clean_vals = [
        attention_diff.clean_stats.entropy,
        attention_diff.clean_stats.sparsity,
        attention_diff.clean_stats.peak_concentration,
        attention_diff.clean_stats.diagonal_ratio,
        attention_diff.clean_stats.head_agreement,
    ]
    attacked_vals = [
        attention_diff.attacked_stats.entropy,
        attention_diff.attacked_stats.sparsity,
        attention_diff.attacked_stats.peak_concentration,
        attention_diff.attacked_stats.diagonal_ratio,
        attention_diff.attacked_stats.head_agreement,
    ]

    x = np.arange(len(stats_names))
    width = 0.35
    axes[1, 0].bar(x - width/2, clean_vals, width, label='Clean', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, attacked_vals, width, label='Attacked', color='red', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(stats_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Attention Statistics Comparison")
    axes[1, 0].legend()

    # Summary text
    summary_text = f"""
Overall Difference Magnitude: {attention_diff.diff_magnitude:.4f}
Entropy Shift: {attention_diff.entropy_shift:+.4f}
Peak Shift: {attention_diff.peak_shift:+.4f}

Affected Layers: {len(attention_diff.affected_layers)} / {n_layers}
Affected Heads: {len(attention_diff.affected_heads)} / {n_layers * clean_attention.shape[1]}

Most Affected Layers: {attention_diff.affected_layers[:5]}
"""
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Summary")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_token_attention_profile(
    analysis: TokenAttentionAnalysis,
    output_path: Path | None = None,
    title: str = "Token Attention Profile",
) -> None:
    """
    Visualize token-level attention analysis.

    Args:
        analysis: TokenAttentionAnalysis object
        output_path: Where to save
        title: Figure title
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available for visualization")
        return

    n_tokens = len(analysis.tokens)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, n_tokens * 0.3), 8))

    # Attention received
    colors_received = ['red' if i in analysis.critical_tokens else
                       ('green' if i == analysis.target_token_idx else 'steelblue')
                       for i in range(n_tokens)]
    axes[0].bar(range(n_tokens), analysis.attention_received, color=colors_received)
    axes[0].set_xlabel("Token Position")
    axes[0].set_ylabel("Attention Received (sum)")
    axes[0].set_title("Attention Received by Each Token")
    axes[0].set_xticks(range(n_tokens))
    axes[0].set_xticklabels(analysis.tokens, rotation=45, ha='right', fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Normal'),
        Patch(facecolor='red', label='Critical (high attention)'),
        Patch(facecolor='green', label='Target'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')

    # Attention given
    axes[1].bar(range(n_tokens), analysis.attention_given, color='steelblue')
    axes[1].set_xlabel("Token Position")
    axes[1].set_ylabel("Attention Given (sum)")
    axes[1].set_title("Attention Given by Each Token")
    axes[1].set_xticks(range(n_tokens))
    axes[1].set_xticklabels(analysis.tokens, rotation=45, ha='right', fontsize=7)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ==============================================================================
# INTEGRATION WITH SCORING
# ==============================================================================

@dataclass
class AdversarialAnalysisResult:
    """Result of adversarial analysis for a test case."""
    test_id: str
    attack_type: str
    match_quality: str  # "NONE", "PARTIAL", "EXACT"

    # Attention analysis (if available)
    attention_stats: AttentionStats | None = None
    attention_diff: AttentionDiff | None = None
    token_analysis: TokenAttentionAnalysis | None = None

    # Degeneration metrics
    degeneration_metrics: dict[str, float] = field(default_factory=dict)

    # Metadata
    expected: str = ""
    actual: str = ""
    prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "test_id": self.test_id,
            "attack_type": self.attack_type,
            "match_quality": self.match_quality,
            "expected": self.expected,
            "actual": self.actual,
            "degeneration_metrics": self.degeneration_metrics,
        }

        if self.attention_stats:
            result["attention_stats"] = self.attention_stats.to_dict()

        if self.attention_diff:
            result["attention_diff"] = self.attention_diff.to_dict()

        if self.token_analysis:
            result["token_analysis"] = {
                "tokens": self.token_analysis.tokens,
                "attention_received": self.token_analysis.attention_received,
                "critical_tokens": self.token_analysis.critical_tokens,
                "distractor_tokens": self.token_analysis.distractor_tokens,
                "target_token_idx": self.token_analysis.target_token_idx,
            }

        return result
