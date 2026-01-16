"""
Model discovery and loading utilities.

Recursively scans directories to find model checkpoints and loads them
for evaluation. Supports various checkpoint formats.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""
    path: Path
    name: str  # Derived model name
    size_bytes: int
    format: str  # 'pytorch', 'safetensors', 'huggingface'
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProtocol(Protocol):
    """Protocol for models that can be evaluated."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        ...

    def get_choice_logprobs(
        self,
        prompt: str,
        choices: list[str],
    ) -> dict[str, float]:
        ...

    def get_attention_weights(
        self,
        prompt: str,
    ) -> Any:
        ...


class CheckpointDiscovery:
    """
    Discovers model checkpoints by recursively walking directories.
    """

    # Common checkpoint file patterns
    CHECKPOINT_PATTERNS = [
        r'.*\.pt$',
        r'.*\.pth$',
        r'.*\.bin$',
        r'.*\.safetensors$',
        r'.*checkpoint.*\.pt$',
        r'.*model.*\.pt$',
    ]

    # Files/directories to skip
    SKIP_PATTERNS = [
        r'optimizer.*',
        r'scheduler.*',
        r'__pycache__',
        r'\.git',
        r'wandb',
    ]

    def __init__(
        self,
        min_size_mb: float = 10.0,
        max_size_gb: float = 50.0,
    ):
        """
        Initialize checkpoint discovery.

        Args:
            min_size_mb: Minimum file size to consider (filters small files)
            max_size_gb: Maximum file size to consider (filters huge files)
        """
        self.min_size_bytes = int(min_size_mb * 1024 * 1024)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

    def discover(self, root_dir: str | Path) -> list[CheckpointInfo]:
        """
        Recursively discover checkpoints in a directory.

        Args:
            root_dir: Root directory to search

        Returns:
            List of discovered checkpoints
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        checkpoints = []

        for path in root_dir.rglob("*"):
            if not path.is_file():
                continue

            # Skip patterns
            if self._should_skip(path):
                continue

            # Check if it matches checkpoint patterns
            if not self._is_checkpoint(path):
                continue

            # Check size
            size = path.stat().st_size
            if size < self.min_size_bytes or size > self.max_size_bytes:
                continue

            # Create checkpoint info
            info = CheckpointInfo(
                path=path,
                name=self._derive_name(path, root_dir),
                size_bytes=size,
                format=self._detect_format(path),
            )

            # Try to extract metadata
            info.metadata = self._extract_metadata(path)

            checkpoints.append(info)

        # Sort by name for consistent ordering
        checkpoints.sort(key=lambda x: x.name)

        return checkpoints

    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        path_str = str(path).lower()
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, path_str):
                return True
        return False

    def _is_checkpoint(self, path: Path) -> bool:
        """Check if file matches checkpoint patterns."""
        name = path.name.lower()
        for pattern in self.CHECKPOINT_PATTERNS:
            if re.match(pattern, name):
                return True
        return False

    def _detect_format(self, path: Path) -> str:
        """Detect checkpoint format from file extension."""
        suffix = path.suffix.lower()
        if suffix == '.safetensors':
            return 'safetensors'
        elif suffix in ['.pt', '.pth']:
            return 'pytorch'
        elif suffix == '.bin':
            return 'huggingface'
        return 'unknown'

    def _derive_name(self, path: Path, root_dir: Path) -> str:
        """
        Derive a readable model name from the checkpoint path.

        Tries to extract meaningful identifiers from the path.
        """
        # Get relative path from root
        try:
            rel_path = path.relative_to(root_dir)
        except ValueError:
            rel_path = path

        # Use parent directory name + file stem
        parts = list(rel_path.parts)

        # Remove common uninformative parts
        skip_parts = {'checkpoints', 'models', 'weights', 'saved'}
        parts = [p for p in parts if p.lower() not in skip_parts]

        if len(parts) >= 2:
            # Use last directory + file stem
            name = f"{parts[-2]}_{path.stem}"
        else:
            name = path.stem

        # Clean up the name
        name = re.sub(r'[_\-]+', '_', name)
        name = name.strip('_')

        # Truncate if too long
        if len(name) > 50:
            name = name[:50]

        return name

    def _extract_metadata(self, path: Path) -> dict[str, Any]:
        """Try to extract metadata from checkpoint."""
        metadata = {
            'size_mb': path.stat().st_size / (1024 * 1024),
        }

        # Try to peek at checkpoint contents
        if path.suffix in ['.pt', '.pth']:
            try:
                # Load only metadata, not full weights
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict):
                    for key in ['config', 'args', 'hparams', 'model_config']:
                        if key in checkpoint:
                            metadata['config'] = checkpoint[key]
                            break
                    if 'step' in checkpoint:
                        metadata['step'] = checkpoint['step']
                    if 'epoch' in checkpoint:
                        metadata['epoch'] = checkpoint['epoch']
                # Don't keep the full checkpoint in memory
                del checkpoint
            except Exception:
                pass

        return metadata


class ModelLoader:
    """
    Loads models from checkpoints for evaluation.
    """

    def __init__(
        self,
        model_class: type | None = None,
        model_config: dict[str, Any] | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer: Any = None,
    ):
        """
        Initialize model loader.

        Args:
            model_class: Class to instantiate for each checkpoint
            model_config: Configuration to pass to model class
            device: Device to load models onto
            tokenizer: Tokenizer to use for all models
        """
        self.model_class = model_class
        self.model_config = model_config or {}
        self.device = device
        self.tokenizer = tokenizer

    def load(self, checkpoint: CheckpointInfo) -> nn.Module | None:
        """
        Load a model from a checkpoint.

        Args:
            checkpoint: Checkpoint info

        Returns:
            Loaded model or None if loading fails
        """
        try:
            print(f"Loading {checkpoint.name} from {checkpoint.path}...")

            if checkpoint.format == 'safetensors':
                return self._load_safetensors(checkpoint)
            elif checkpoint.format == 'pytorch':
                return self._load_pytorch(checkpoint)
            elif checkpoint.format == 'huggingface':
                return self._load_huggingface(checkpoint)
            else:
                print(f"  Unknown format: {checkpoint.format}")
                return None

        except Exception as e:
            print(f"  Failed to load: {e}")
            return None

    def _load_pytorch(self, checkpoint: CheckpointInfo) -> nn.Module | None:
        """Load PyTorch checkpoint."""
        state = torch.load(checkpoint.path, map_location=self.device, weights_only=False)

        # Handle different checkpoint structures
        if isinstance(state, dict):
            # Look for model state dict
            for key in ['model', 'state_dict', 'model_state_dict', 'module']:
                if key in state:
                    state = state[key]
                    break

        if self.model_class is None:
            print(f"  No model class provided, returning state dict")
            return state

        # Instantiate model and load weights
        model = self.model_class(**self.model_config)
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()

        return model

    def _load_safetensors(self, checkpoint: CheckpointInfo) -> nn.Module | None:
        """Load safetensors checkpoint."""
        try:
            from safetensors.torch import load_file
        except ImportError:
            print("  safetensors not installed")
            return None

        state = load_file(checkpoint.path, device=self.device)

        if self.model_class is None:
            return state

        model = self.model_class(**self.model_config)
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()

        return model

    def _load_huggingface(self, checkpoint: CheckpointInfo) -> nn.Module | None:
        """Load HuggingFace checkpoint."""
        # For .bin files, usually part of a HF model directory
        parent_dir = checkpoint.path.parent

        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                parent_dir,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            model.eval()
            return model
        except Exception as e:
            print(f"  Failed to load as HuggingFace model: {e}")
            return None

    def load_all(
        self,
        checkpoints: list[CheckpointInfo],
    ) -> dict[str, nn.Module]:
        """
        Load all checkpoints.

        Args:
            checkpoints: List of checkpoints to load

        Returns:
            Dict mapping model name to loaded model
        """
        models = {}

        for cp in checkpoints:
            model = self.load(cp)
            if model is not None:
                models[cp.name] = model
                print(f"  Loaded: {cp.name}")

        return models


class EvaluableModel:
    """
    Wrapper that makes a PyTorch model compatible with the evaluation protocol.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda",
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        # Ensure model is in eval mode
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        """Generate text continuation."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
        ).to(self.device)

        with torch.no_grad():
            if temperature == 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

        # Decode only the new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )

        return generated

    def get_choice_logprobs(
        self,
        prompt: str,
        choices: list[str],
    ) -> dict[str, float]:
        """Get log probabilities for each choice."""
        logprobs = {}

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get logits for the last position
            last_logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(last_logits, dim=-1)

        for choice in choices:
            # Tokenize just the choice
            choice_ids = self.tokenizer.encode(choice, add_special_tokens=False)
            if choice_ids:
                # Use first token of choice
                token_id = choice_ids[0]
                logprobs[choice] = log_probs[token_id].item()

        return logprobs

    def get_attention_weights(self, prompt: str) -> Any:
        """Get attention weights for visualization."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        if hasattr(outputs, 'attentions') and outputs.attentions:
            # Stack attention from all layers: [layers, batch, heads, seq, seq]
            attention = torch.stack(outputs.attentions, dim=0)
            # Remove batch dimension: [layers, heads, seq, seq]
            return attention[:, 0, :, :, :].cpu().numpy()

        return None


def discover_and_load_models(
    checkpoint_dir: str | Path,
    model_class: type | None = None,
    model_config: dict[str, Any] | None = None,
    tokenizer: Any = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    min_size_mb: float = 10.0,
) -> dict[str, EvaluableModel]:
    """
    Convenience function to discover and load all models from a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints
        model_class: Class to instantiate for each checkpoint
        model_config: Configuration for model class
        tokenizer: Tokenizer to use
        device: Device to load onto
        min_size_mb: Minimum checkpoint size

    Returns:
        Dictionary mapping model names to EvaluableModel instances
    """
    # Discover checkpoints
    discovery = CheckpointDiscovery(min_size_mb=min_size_mb)
    checkpoints = discovery.discover(checkpoint_dir)

    print(f"Discovered {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name}: {cp.path} ({cp.metadata.get('size_mb', 0):.1f} MB)")

    if not checkpoints:
        return {}

    # Load models
    loader = ModelLoader(
        model_class=model_class,
        model_config=model_config,
        device=device,
        tokenizer=tokenizer,
    )

    raw_models = loader.load_all(checkpoints)

    # Wrap in EvaluableModel
    evaluable_models = {}
    for name, model in raw_models.items():
        if isinstance(model, nn.Module) and tokenizer is not None:
            evaluable_models[name] = EvaluableModel(model, tokenizer, device)
        else:
            # Already wrapped or state dict
            evaluable_models[name] = model

    return evaluable_models
