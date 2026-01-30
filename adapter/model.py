from __future__ import annotations

from typing import Any
import torch
from torch import nn
from transformers import PretrainedConfig
from console import logger
from model import Model


class HFConfigShim(PretrainedConfig):
    """Minimal config shim based on PretrainedConfig for PEFT/Trainer compatibility."""
    
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 22,
        vocab_size: int = 50304,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs: Any,
    ):
        # Initialize PretrainedConfig first (may be called by serialization/deserialization)
        # Extract our custom args from kwargs if they were passed that way
        hidden_size = kwargs.pop("hidden_size", hidden_size)
        num_attention_heads = kwargs.pop("num_attention_heads", num_attention_heads)
        num_hidden_layers = kwargs.pop("num_hidden_layers", num_hidden_layers)
        vocab_size = kwargs.pop("vocab_size", vocab_size)
        pad_token_id = kwargs.pop("pad_token_id", pad_token_id)
        eos_token_id = kwargs.pop("eos_token_id", eos_token_id)
        
        super().__init__(**kwargs)
        # Set our custom attributes
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class CompatibleWrapper(torch.nn.Module):
    """Wraps Caramba Model to be compatible with PEFT and HuggingFace Trainer.
    
    This wrapper ensures that PEFT can find `base_model` and `config` attributes
    directly, which is required for proper attribute resolution.
    """
    
    def __init__(self, base_model: Model, config: PretrainedConfig):
        super().__init__()
        # Register base_model as a single submodule.
        #
        # IMPORTANT: Do not re-register submodules that already live under base_model
        # (e.g. base_model.topology/embedder) as *additional* children of this wrapper.
        # Doing so creates duplicate module references in the module tree, which can
        # break FSDP auto-wrapping (it asserts if it encounters an already-wrapped
        # child via a second reference).
        self.base_model = base_model
        # Explicitly set config as direct attribute (required by Trainer/PEFT)
        self.config = config
        
        # Verify base_model is not None
        if self.base_model is None:
            raise ValueError("base_model cannot be None")
        
        # Verify we can access input embeddings (non-fatal check)
        # PEFT will handle missing embeddings with better error messages
        try:
            emb = self.get_input_embeddings()
            if emb is None:
                logger.warning("get_input_embeddings() returned None - PEFT may handle this")
        except Exception as e:
            logger.warning(f"Could not verify input embeddings in __init__ (non-fatal): {e}")

    # Expose HF-style backbone attributes for PEFT without duplicating module registration.
    @property
    def transformer(self) -> nn.Module:
        if not hasattr(self.base_model, "topology"):
            raise AttributeError("base_model has no attribute 'topology' (expected transformer backbone)")
        return self.base_model.topology

    @property
    def embedder(self) -> nn.Module:
        if not hasattr(self.base_model, "embedder"):
            raise AttributeError("base_model has no attribute 'embedder'")
        return self.base_model.embedder
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model (for PEFT/Trainer compatibility)."""
        return self.base_model.device
    
    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Delegate named_parameters to base_model for PEFT compatibility."""
        return self.base_model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def parameters(self, recurse: bool = True):
        """Delegate parameters to base_model for PEFT compatibility."""
        return self.base_model.parameters(recurse=recurse)
    
    def get_input_embeddings(self):
        """Get input embeddings (for PEFT compatibility).
        
        PEFT requires this to return a valid nn.Module, not None.
        """
        # Try the base model's method first
        if hasattr(self.base_model, "get_input_embeddings"):
            emb = self.base_model.get_input_embeddings()
            if emb is not None:
                return emb
        
        # Fallback: try to get embedder's token_embedding
        if hasattr(self.base_model, "embedder"):
            emb = getattr(self.base_model.embedder, "token_embedding", None)
            if emb is not None:
                return emb
        
        # If we can't find embeddings, raise an error rather than returning None
        # PEFT cannot work without input embeddings
        raise RuntimeError(
            "Could not find input embeddings for PEFT. "
            "Model must have get_input_embeddings() or embedder.token_embedding"
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, object]:
        """Prepare inputs for generation (for PEFT compatibility).
        
        This is a HuggingFace-style method that PEFT expects. It ensures
        inputs are on the correct device and returns them in a dict format.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional generation arguments (attention_mask, past_key_values, etc.)
            
        Returns:
            Dictionary with prepared inputs
        """
        # Ensure input_ids are on the model's device
        model_device = self.device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        
        # Build return dict with input_ids and any other kwargs
        # Type as dict[str, object] to allow both tensors and other objects
        prepared: dict[str, object] = {}
        prepared["input_ids"] = input_ids
        
        # Move other tensor kwargs to the correct device
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.device != model_device:
                prepared[key] = value.to(model_device)
            else:
                prepared[key] = value
        
        return prepared
    
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Forward pass compatible with HuggingFace Trainer.
        
        Args:
            input_ids: Token IDs, shape (B, T)
            attention_mask: Optional attention mask (ignored for now)
            labels: Optional labels for loss computation, shape (B, T)
            **kwargs: Additional arguments (may include input_ids/labels if passed as dict)
            
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Handle case where data is passed as kwargs dict (from data collator)
        # PEFT p-tuning may provide inputs_embeds instead of input_ids
        inputs_embeds: torch.Tensor | None = None
        # Caramba KV-cache context (InferContext) can be threaded through PEFT/HF wrappers
        # via kwargs. If present, pass it down to the underlying Caramba model/topology.
        ctx = kwargs.pop("ctx", None)
        if "inputs_embeds" in kwargs:
            val = kwargs.pop("inputs_embeds")
            if isinstance(val, torch.Tensor):
                inputs_embeds = val
        
        if input_ids is None:
            if "input_ids" in kwargs:
                val = kwargs.pop("input_ids")
                if isinstance(val, torch.Tensor):
                    input_ids = val
            elif len(kwargs) == 1:
                first_val = list(kwargs.values())[0]
                if isinstance(first_val, dict):
                    # Handle case where batch dict is passed as single kwarg
                    batch = first_val
                    val = batch.get("input_ids") if isinstance(batch, dict) else None
                    if isinstance(val, torch.Tensor):
                        input_ids = val
                    if inputs_embeds is None:
                        emb_val = batch.get("inputs_embeds") if isinstance(batch, dict) else None
                        if isinstance(emb_val, torch.Tensor):
                            inputs_embeds = emb_val
                    if labels is None:
                        label_val = batch.get("labels") if isinstance(batch, dict) else None
                        if isinstance(label_val, torch.Tensor):
                            labels = label_val
        
        # If inputs_embeds is provided (from PEFT p-tuning), use it directly
        if inputs_embeds is not None:
            # Bypass embedder and call topology directly, then compute logits
            features = self.base_model.topology(inputs_embeds, ctx=ctx)  # type: ignore[call-arg]
            logits = self.base_model._features_to_logits(features)
        elif input_ids is not None:
            # Normal path: use input_ids, model will embed internally
            logits = self.base_model(input_ids, ctx=ctx)  # type: ignore[call-arg]
        else:
            raise ValueError(
                f"Either input_ids or inputs_embeds required for training. Got kwargs keys: {list(kwargs.keys())}"
            )
        
        out = {"logits": logits}
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            out["loss"] = loss
        
        return out
