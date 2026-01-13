"""System module

The system module contains the system for the manifest.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, StrictBool

from caramba.manifest.embedder import Embedder
from caramba.manifest.topology import Topology


class SystemType(str, Enum):
    """The type of system."""
    LANGUAGE_MODEL = "language_model"
    GENERIC = "generic"
    MLP_CLASSIFIER = "mlp_classifier"
    GCN = "gcn"
    DIFFUSION_DENOISER = "diffusion_denoiser"
    DIFFUSION_CODEGEN = "diffusion_codegen"
    CCL = "ccl"


class ModelType(str, Enum):
    """Model type enumeration."""
    TRANSFORMER = "transformer"
    GPT = "gpt"
    VIT = "vit"
    MLP = "mlp"


class SystemModel(BaseModel):
    """The model for the system."""
    type: ModelType
    tied_embeddings: StrictBool
    embedder: Embedder
    topology: Topology


class System(BaseModel):
    """The system for the manifest."""
    type: SystemType
    name: str
    description: str
    model: SystemModel