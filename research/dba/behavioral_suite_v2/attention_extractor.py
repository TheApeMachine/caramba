"""
Attention extraction utilities for transformer models.

Provides hook-based attention capture compatible with:
- HuggingFace transformers
- Custom PyTorch models with standard attention modules

Usage:
    model = AutoModelForCausalLM.from_pretrained(...)
    extractor = AttentionExtractor(model)

    with extractor.capture():
        output = model.generate(input_ids)

    attention = extractor.get_attention()  # [layers, heads, seq, seq]
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


class AttentionExtractor:
    """
    Extracts attention weights from transformer models using forward hooks.

    Supports both standard HuggingFace models and custom implementations
    that use nn.MultiheadAttention or similar modules.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_module_names: list[str] | None = None,
        layer_pattern: str = "layers.{}.attention",
    ):
        """
        Initialize attention extractor.

        Args:
            model: The transformer model
            attention_module_names: Explicit list of module names to hook.
                If None, auto-detects based on layer_pattern.
            layer_pattern: Pattern for finding attention modules.
                Use {} as placeholder for layer index.
        """
        self.model = model
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.attention_weights: list[torch.Tensor] = []
        self.layer_pattern = layer_pattern

        # Find attention modules
        if attention_module_names:
            self.attention_modules = attention_module_names
        else:
            self.attention_modules = self._find_attention_modules()

    def _find_attention_modules(self) -> list[str]:
        """Auto-detect attention modules in the model."""
        modules = []

        # Common patterns for attention modules
        patterns = [
            "attn",
            "attention",
            "self_attn",
            "self_attention",
        ]

        for name, module in self.model.named_modules():
            name_lower = name.lower()
            if any(p in name_lower for p in patterns):
                # Check if this module computes attention weights
                if hasattr(module, 'forward'):
                    modules.append(name)

        return modules

    def _get_module(self, name: str) -> nn.Module:
        """Get module by name from model."""
        parts = name.split('.')
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module

    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that captures attention weights."""
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # Handle different output formats
            if isinstance(output, tuple):
                # Many attention modules return (output, attention_weights)
                for item in output:
                    if isinstance(item, torch.Tensor) and len(item.shape) >= 3:
                        # Check if it looks like attention weights
                        if item.shape[-1] == item.shape[-2]:
                            self.attention_weights.append(
                                item.detach().cpu()
                            )
                            return
            elif isinstance(output, dict):
                # Some modules return a dict with 'attn_weights' key
                if 'attn_weights' in output:
                    self.attention_weights.append(
                        output['attn_weights'].detach().cpu()
                    )
            elif hasattr(module, 'attention_weights'):
                # Some modules store attention as an attribute
                attn_weights = getattr(module, 'attention_weights', None)
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    self.attention_weights.append(
                        attn_weights.detach().cpu()
                    )

        return hook

    @contextmanager
    def capture(self):
        """
        Context manager for capturing attention during forward pass.

        Usage:
            with extractor.capture():
                model(input_ids)
            attention = extractor.get_attention()
        """
        self.attention_weights = []

        # Register hooks
        for i, name in enumerate(self.attention_modules):
            try:
                module = self._get_module(name)
                hook = module.register_forward_hook(self._create_hook(i))
                self.hooks.append(hook)
            except AttributeError:
                continue

        try:
            yield
        finally:
            # Remove hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []

    def get_attention(self) -> np.ndarray | None:
        """
        Get captured attention weights.

        Returns:
            Attention tensor of shape [layers, heads, seq_len, seq_len]
            or None if no attention was captured.
        """
        if not self.attention_weights:
            return None

        # Stack attention from all layers
        try:
            # Handle different shapes
            processed = []
            for attn in self.attention_weights:
                if len(attn.shape) == 4:
                    # [batch, heads, seq, seq] -> [heads, seq, seq]
                    processed.append(attn[0].numpy())
                elif len(attn.shape) == 3:
                    # [heads, seq, seq]
                    processed.append(attn.numpy())

            if not processed:
                return None

            # Stack into [layers, heads, seq, seq]
            return np.stack(processed, axis=0)

        except Exception as e:
            print(f"Warning: Failed to process attention: {e}")
            return None

    def get_layer_attention(self, layer: int) -> np.ndarray | None:
        """Get attention for a specific layer."""
        attention = self.get_attention()
        if attention is None or layer >= attention.shape[0]:
            return None
        return attention[layer]


class HuggingFaceAttentionExtractor(AttentionExtractor):
    """
    Specialized extractor for HuggingFace transformers models.

    These models often have a standard structure that we can exploit.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._setup_hf_hooks()

    def _setup_hf_hooks(self) -> None:
        """Set up hooks specific to HuggingFace model structure."""
        # Try to detect model type and configure accordingly
        model_type = getattr(self.model.config, 'model_type', None)

        if model_type == 'llama':
            num_layers = getattr(self.model.config, 'num_hidden_layers', None)
            if num_layers is not None:
                num_layers_int: int = int(num_layers)  # type: ignore[reportArgumentType]
                self.attention_modules = [
                    f'model.layers.{i}.self_attn'
                    for i in range(num_layers_int)
                ]
        elif model_type == 'gpt2':
            n_layer = getattr(self.model.config, 'n_layer', None)
            if n_layer is not None:
                n_layer_int: int = int(n_layer)  # type: ignore[reportArgumentType]
                self.attention_modules = [
                    f'transformer.h.{i}.attn'
                    for i in range(n_layer_int)
                ]
        elif model_type == 'gpt_neo':
            num_layers = getattr(self.model.config, 'num_layers', None)
            if num_layers is not None:
                num_layers_int: int = int(num_layers)  # type: ignore[reportArgumentType]
                self.attention_modules = [
                    f'transformer.h.{i}.attn.attention'
                    for i in range(num_layers_int)
                ]
        # Add more model types as needed


def extract_attention_for_prompt(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    device: str = "cuda",
) -> np.ndarray | None:
    """
    Convenience function to extract attention for a single prompt.

    Args:
        model: The transformer model
        tokenizer: Tokenizer for the model
        prompt: Input text
        device: Device to run on

    Returns:
        Attention tensor [layers, heads, seq_len, seq_len]
    """
    extractor = AttentionExtractor(model)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Capture attention
    with extractor.capture():
        with torch.no_grad():
            model(**inputs, output_attentions=True)

    return extractor.get_attention()


def normalize_attention(
    attention: np.ndarray,
    method: str = "softmax",
) -> np.ndarray:
    """
    Normalize attention weights for visualization.

    Args:
        attention: Raw attention weights
        method: Normalization method ('softmax', 'minmax', 'none')

    Returns:
        Normalized attention
    """
    if method == "softmax":
        # Already normalized via softmax during forward pass
        return attention
    elif method == "minmax":
        # Scale to [0, 1]
        vmin, vmax = attention.min(), attention.max()
        if vmax - vmin > 1e-8:
            return (attention - vmin) / (vmax - vmin)
        return attention
    else:
        return attention


def compute_attention_stats(attention: np.ndarray) -> dict[str, float]:
    """
    Compute statistics about attention patterns.

    Args:
        attention: Attention tensor [layers, heads, seq, seq]

    Returns:
        Dictionary of statistics
    """
    # Entropy (measure of uniformity)
    eps = 1e-10
    entropy = -np.sum(attention * np.log(attention + eps), axis=-1)
    avg_entropy = entropy.mean()

    # Sparsity (fraction near zero)
    sparsity = (attention < 0.01).mean()

    # Peak position stats
    peak_positions = attention.argmax(axis=-1)
    diagonal_ratio = (peak_positions == np.arange(attention.shape[-1])).mean()

    return {
        "avg_entropy": float(avg_entropy),
        "sparsity": float(sparsity),
        "diagonal_ratio": float(diagonal_ratio),
        "max_attention": float(attention.max()),
        "min_attention": float(attention.min()),
    }
