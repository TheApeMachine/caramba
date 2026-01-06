"""DBA memory summarization helpers

When sequences get long, DBA can compress “remote” history into a small number
of summary tokens while keeping semantic and geometric paths aligned, so both
channels continue to refer to the same compressed context.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from caramba.config.layer import AttentionLayerConfig


class DecoupledMemorySummarizer:
    """DBA memory summarization mixin

    A mixin keeps summarization logic separate from attention math; the main
    layer just calls into `maybe_summarize_kv_decoupled` when configured.
    """

    config: AttentionLayerConfig
    _v_head_dim: int
    mem_k_proj_sem: nn.Module | None
    mem_k_proj_geo: nn.Module | None
    mem_v_proj_dba: nn.Module | None
    mem_k_proj: nn.Module | None
    mem_v_proj: nn.Module | None

    def _init_memory_summarizer_decoupled(self) -> None:
        """Initialize DBA memory summarizer modules

        Linear/conv summarizers start as “do nothing” equivalents, so you can
        turn them on without immediately changing behavior in surprising ways.
        """
        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        sem_head_dim = int(self.config.sem_head_dim or 0)
        geo_head_dim = int(self.config.geo_head_dim or 0)
        v_head_dim = int(getattr(self, "_v_head_dim", 0) or 0)

        if kind == "linear":
            self._init_linear(sem_head_dim=sem_head_dim, geo_head_dim=geo_head_dim, v_head_dim=v_head_dim)
        elif kind == "conv":
            self._init_conv(sem_head_dim=sem_head_dim, geo_head_dim=geo_head_dim, v_head_dim=v_head_dim)
        else:
            self.mem_k_proj_sem = None
            self.mem_k_proj_geo = None
            self.mem_v_proj_dba = None

        self.mem_k_proj = None
        self.mem_v_proj = None

    def _init_linear(self, *, sem_head_dim: int, geo_head_dim: int, v_head_dim: int) -> None:
        if sem_head_dim <= 0 or geo_head_dim <= 0 or v_head_dim <= 0:
            self.mem_k_proj_sem = None
            self.mem_k_proj_geo = None
            self.mem_v_proj_dba = None
            return
        self.mem_k_proj_sem = nn.Linear(sem_head_dim, sem_head_dim, bias=False)
        self.mem_k_proj_geo = nn.Linear(geo_head_dim, geo_head_dim, bias=False)
        self.mem_v_proj_dba = nn.Linear(v_head_dim, v_head_dim, bias=False)
        nn.init.eye_(cast(nn.Linear, self.mem_k_proj_sem).weight)
        nn.init.eye_(cast(nn.Linear, self.mem_k_proj_geo).weight)
        nn.init.eye_(cast(nn.Linear, self.mem_v_proj_dba).weight)

    def _init_conv(self, *, sem_head_dim: int, geo_head_dim: int, v_head_dim: int) -> None:
        if sem_head_dim <= 0 or geo_head_dim <= 0 or v_head_dim <= 0:
            self.mem_k_proj_sem = None
            self.mem_k_proj_geo = None
            self.mem_v_proj_dba = None
            return
        self.mem_k_proj_sem = nn.Conv1d(sem_head_dim, sem_head_dim, kernel_size=3, padding=1, groups=sem_head_dim, bias=False)
        self.mem_k_proj_geo = nn.Conv1d(geo_head_dim, geo_head_dim, kernel_size=3, padding=1, groups=geo_head_dim, bias=False)
        self.mem_v_proj_dba = nn.Conv1d(v_head_dim, v_head_dim, kernel_size=3, padding=1, groups=v_head_dim, bias=False)
        # Initialize as identity/pass-through (do nothing): center weight = 1, others = 0.
        w = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        for m in (self.mem_k_proj_sem, self.mem_k_proj_geo, self.mem_v_proj_dba):
            ww = cast(nn.Conv1d, m).weight
            d = int(ww.shape[0])
            ww.data.zero_()
            ww.data[:, 0, :].copy_(w.to(device=ww.device, dtype=ww.dtype).view(1, 3).expand(d, 3))

    def _conv1d_path(self, *, x: Tensor, mod: nn.Module | None, BH: int, remote_len: int) -> Tensor:
        if mod is None:
            return x
        d = int(x.size(-1))
        xin = x.reshape(BH, remote_len, d).transpose(1, 2)
        y = cast(nn.Conv1d, mod)(xin).transpose(1, 2).reshape_as(x)
        return y

    def maybe_summarize_kv_decoupled(
        self,
        *,
        k_sem: Tensor,
        k_geo: Tensor,
        v: Tensor,
        k_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Summarize older KV blocks for DBA

        The summarizer compresses old tokens into block means (optionally with a
        learned linear/conv transform), which reduces KV length without
        discarding the newest tokens.
        """
        mem_block = getattr(self.config, "mem_block", None)
        if mem_block is None:
            return k_sem, k_geo, v, k_pos
        mb = int(mem_block)
        if mb <= 0:
            return k_sem, k_geo, v, k_pos
        if k_sem.size(2) == 0:
            return k_sem, k_geo, v, k_pos

        threshold = getattr(self.config, "mem_activation_threshold", None)
        if threshold is not None and int(k_sem.size(2)) < int(threshold):
            return k_sem, k_geo, v, k_pos

        local_window = getattr(self.config, "local_window", None)
        lw = int(local_window) if local_window is not None else 0
        T = int(k_sem.size(2))
        if lw <= 0 or lw >= T:
            return k_sem, k_geo, v, k_pos
        remote_len = T - lw
        if remote_len <= 0:
            return k_sem, k_geo, v, k_pos

        ks_r, ks_l = k_sem[:, :, :remote_len, :], k_sem[:, :, remote_len:, :]
        kg_r, kg_l = k_geo[:, :, :remote_len, :], k_geo[:, :, remote_len:, :]
        v_r, v_l = v[:, :, :remote_len, :], v[:, :, remote_len:, :]
        pos_r, pos_l = k_pos[:remote_len], k_pos[remote_len:]

        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        if kind == "conv":
            BH = int(ks_r.size(0) * ks_r.size(1))
            ks_r = self._conv1d_path(x=ks_r, mod=self.mem_k_proj_sem, BH=BH, remote_len=remote_len)
            kg_r = self._conv1d_path(x=kg_r, mod=self.mem_k_proj_geo, BH=BH, remote_len=remote_len)
            v_r = self._conv1d_path(x=v_r, mod=self.mem_v_proj_dba, BH=BH, remote_len=remote_len)

        B0, H0, _Tr, _ = ks_r.shape
        n_full = remote_len // mb
        rem = remote_len - n_full * mb

        ks_full = ks_r.new_empty((B0, H0, 0, ks_r.size(-1)))
        kg_full = kg_r.new_empty((B0, H0, 0, kg_r.size(-1)))
        v_full = v_r.new_empty((B0, H0, 0, v_r.size(-1)))
        pos_full = pos_r[:0]
        if n_full > 0:
            ks_full = ks_r[:, :, : n_full * mb, :].reshape(B0, H0, n_full, mb, ks_r.size(-1)).mean(dim=3)
            kg_full = kg_r[:, :, : n_full * mb, :].reshape(B0, H0, n_full, mb, kg_r.size(-1)).mean(dim=3)
            v_full = v_r[:, :, : n_full * mb, :].reshape(B0, H0, n_full, mb, v_r.size(-1)).mean(dim=3)
            pos_full = pos_r[(mb - 1) : (n_full * mb) : mb]

        if rem > 0:
            ks_tail = ks_r[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)
            kg_tail = kg_r[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)
            v_tail = v_r[:, :, n_full * mb : remote_len, :].mean(dim=2, keepdim=True)
            pos_tail = pos_r[remote_len - 1 : remote_len]
            ks_mem = torch.cat([ks_full, ks_tail], dim=2)
            kg_mem = torch.cat([kg_full, kg_tail], dim=2)
            v_mem = torch.cat([v_full, v_tail], dim=2)
            pos_mem = torch.cat([pos_full, pos_tail], dim=0)
        else:
            ks_mem, kg_mem, v_mem, pos_mem = ks_full, kg_full, v_full, pos_full

        if kind == "linear":
            if self.mem_k_proj_sem is not None:
                ks_mem = cast(nn.Linear, self.mem_k_proj_sem)(ks_mem)
            if self.mem_k_proj_geo is not None:
                kg_mem = cast(nn.Linear, self.mem_k_proj_geo)(kg_mem)
            if self.mem_v_proj_dba is not None:
                v_mem = cast(nn.Linear, self.mem_v_proj_dba)(v_mem)

        ks2 = torch.cat([ks_mem, ks_l], dim=2)
        kg2 = torch.cat([kg_mem, kg_l], dim=2)
        v2 = torch.cat([v_mem, v_l], dim=2)
        pos2 = torch.cat([pos_mem, pos_l], dim=0)
        return ks2, kg2, v2, pos2

