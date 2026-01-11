"""Constructive Compression Learning (CCL).

This package provides a non-gradient learning pipeline:
- learn a discrete codec (vector quantization) over local patches
- tokenize inputs into a discrete grid
- fit simple probabilistic context models by counting with Dirichlet smoothing

The goal is to make the technique manifest-driven and architecture-agnostic:
CCL can be used as a trainer/system inside Caramba targets without assuming
transformers, SGD, or any specific dataset.
"""

