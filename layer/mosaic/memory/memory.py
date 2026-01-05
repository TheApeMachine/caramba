"""MOSAIC memory

Hard-addressed, set-associative memory with tag-driven routing and fuzzy in-bucket match.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from caramba.config.layer import MosaicBlockLayerConfig
from caramba.layer.mosaic.memory.reader import MemoryReader
from caramba.layer.mosaic.memory.routing import BitRouter, VqRouter
from caramba.layer.mosaic.memory.vsa import VsaNovelty, VsaTagProjector
from caramba.layer.mosaic.memory.writer import MemoryWriter
from caramba.layer.mosaic.state import MosaicState


class MosaicMemory(nn.Module):
    """MOSAIC memory subsystem."""

    def __init__(self, config: MosaicBlockLayerConfig, d_model: int) -> None:
        super().__init__()
        self.config = config
        self.mem_buckets = int(config.mem_buckets)
        self.mem_hashes = int(config.mem_hashes)
        self.mem_dim = int(config.mem_dim)
        self.mem_assoc = int(getattr(config, "mem_assoc", 1))
        self.mem_key_dim = int(getattr(config, "mem_key_dim", 32))
        self.mem_router = str(getattr(config, "mem_router", "bits")).lower().strip()
        self.mem_read_temp = float(getattr(config, "mem_read_temp", 1.0))
        self.mem_write_threshold = float(getattr(config, "mem_write_threshold", 0.5))
        self.mem_write_eta = float(getattr(config, "mem_write_eta", 0.1))
        self.mem_match_threshold = float(getattr(config, "mem_match_threshold", 0.0))
        self.mem_vsa_enabled = bool(getattr(config, "mem_vsa_enabled", True))
        self.mem_vsa_dim = int(getattr(config, "mem_vsa_dim", 32))
        self.mem_vsa_weight = float(getattr(config, "mem_vsa_weight", 1.0))
        self.mem_vsa_tanh_scale = float(getattr(config, "mem_vsa_tanh_scale", 1.0))
        self.mem_vsa_novelty_beta = float(getattr(config, "mem_vsa_novelty_beta", 1.0))
        self.mem_vsa_novelty_threshold = float(getattr(config, "mem_vsa_novelty_threshold", 0.0))

        if self.mem_buckets < 2:
            raise ValueError("mem_buckets must be >= 2")
        if self.mem_hashes < 1:
            raise ValueError("mem_hashes must be >= 1")
        if self.mem_dim < 1:
            raise ValueError("mem_dim must be >= 1")
        if self.mem_assoc < 1:
            raise ValueError("mem_assoc must be >= 1")
        if self.mem_key_dim < 1:
            raise ValueError("mem_key_dim must be >= 1")
        if self.mem_vsa_dim < 1:
            raise ValueError("mem_vsa_dim must be >= 1")
        if self.mem_vsa_weight < 0.0:
            raise ValueError("mem_vsa_weight must be >= 0")
        if self.mem_vsa_tanh_scale <= 0.0:
            raise ValueError("mem_vsa_tanh_scale must be > 0")
        if self.mem_vsa_novelty_beta <= 0.0:
            raise ValueError("mem_vsa_novelty_beta must be > 0")

        self.mem_qkey = nn.Linear(int(d_model), int(self.mem_key_dim), bias=False)
        self.mem_wkey = nn.Linear(int(d_model), int(self.mem_key_dim), bias=False)
        self.mem_value = nn.Linear(int(d_model), int(self.mem_dim), bias=False)
        self.mem_out = nn.Linear(int(self.mem_dim), int(d_model), bias=False)
        self.mem_write_gate = nn.Linear(int(d_model), 1, bias=True)
        self.mem_utility_head = nn.Linear(int(d_model), 1, bias=True)

        self.vsa_projector: VsaTagProjector | None = None
        self.vsa_novelty: VsaNovelty | None = None
        if self.mem_vsa_enabled:
            self.vsa_projector = VsaTagProjector(
                in_dim=int(self.mem_key_dim),
                vsa_dim=int(self.mem_vsa_dim),
                tanh_scale=float(self.mem_vsa_tanh_scale),
            )
            self.vsa_novelty = VsaNovelty(
                beta=float(self.mem_vsa_novelty_beta),
                threshold=float(self.mem_vsa_novelty_threshold),
            )

        self.bit_router: BitRouter | None = None
        self.vq_router: VqRouter | None = None
        if self.mem_router == "bits":
            self.bit_router = BitRouter(in_dim=int(self.mem_key_dim), hashes=int(self.mem_hashes), buckets=int(self.mem_buckets))
        else:
            self.vq_router = VqRouter(
                in_dim=int(self.mem_key_dim),
                hashes=int(self.mem_hashes),
                buckets=int(self.mem_buckets),
                groups=int(getattr(self.config, "mem_vq_groups", 2)),
                codebook_size=int(getattr(self.config, "mem_vq_codebook_size", 256)),
                group_dim=int(getattr(self.config, "mem_vq_group_dim", 16)),
                beam=int(getattr(self.config, "mem_vq_beam", 1)),
                write_multi=bool(getattr(self.config, "mem_write_multi", False)),
            )
        self.reader = MemoryReader(
            mem_out=self.mem_out,
            mem_qkey=self.mem_qkey,
            mem_key_dim=int(self.mem_key_dim),
            mem_dim=int(self.mem_dim),
            mem_tag_dim=int(self.mem_vsa_dim),
            mem_assoc=int(self.mem_assoc),
            mem_hashes=int(self.mem_hashes),
            mem_buckets=int(self.mem_buckets),
            mem_read_temp=float(self.mem_read_temp),
            mem_vsa_weight=float(self.mem_vsa_weight),
            mem_vsa_enabled=bool(self.mem_vsa_enabled),
            vsa_projector=self.vsa_projector,
        )
        self.writer = MemoryWriter(
            mem_wkey=self.mem_wkey,
            mem_value=self.mem_value,
            mem_write_gate=self.mem_write_gate,
            mem_buckets=int(self.mem_buckets),
            mem_hashes=int(self.mem_hashes),
            mem_assoc=int(self.mem_assoc),
            mem_key_dim=int(self.mem_key_dim),
            mem_tag_dim=int(self.mem_vsa_dim),
            mem_dim=int(self.mem_dim),
            mem_write_threshold=float(self.mem_write_threshold),
            mem_write_eta=float(self.mem_write_eta),
            mem_match_threshold=float(self.mem_match_threshold),
            mem_vsa_enabled=bool(self.mem_vsa_enabled),
            vsa_projector=self.vsa_projector,
            vsa_novelty=self.vsa_novelty,
        )

    def compute_routing(self, u: Tensor, *, collect_aux: bool) -> dict[str, Any]:
        """Compute routing indices for read/write for full sequence."""
        tag = self.mem_qkey(u).to(dtype=u.dtype)
        if self.mem_router == "bits":
            if self.bit_router is None:
                raise RuntimeError("bit_router is None but mem_router='bits'")
            routing = self.bit_router.route(tag=tag, collect_aux=collect_aux)
        else:
            if self.vq_router is None:
                raise RuntimeError("vq_router is None but mem_router='vq'")
            routing = self.vq_router.route(tag=tag, collect_aux=collect_aux)
        out: dict[str, Any] = {"idx_r": routing.idx_r, "idx_w": routing.idx_w}
        out.update(routing.aux)
        return out

    def read(self, u: Tensor, st: MosaicState, routing: dict[str, Any]) -> Tensor:
        return self.reader.read(u, st, routing)

    def write_chunk(
        self,
        u: Tensor,
        st: MosaicState,
        routing: dict[str, Any],
        t0: int,
        mask: Tensor | None,
        *,
        write_scale: Tensor | None = None,
    ) -> Tensor:
        return self.writer.write_chunk(u, st, routing, int(t0), mask, write_scale)

