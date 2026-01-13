"""Layer stats manager.

Attaches forward hooks to model modules and aggregates lightweight statistics at
configurable intervals.
"""

from __future__ import annotations

import traceback
from typing import Any

import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from caramba.collector.layer.stats import LayerStatsCollector
from caramba.console import logger
from caramba.layer.attention import AttentionLayer
from caramba.layer.memory_block.block import MemoryBlockLayer


class LayerStatsManager:
    def __init__(self, *, system: object, interval: int) -> None:
        self.interval = max(0, int(interval))
        self.enabled = self.interval > 0

        self.attn_modules: list[tuple[int, str, nn.Module]] = []
        self.mosaic_modules: list[tuple[int, str, nn.Module]] = []
        self._handles: list[RemovableHandle] = []
        self._collector = LayerStatsCollector(self.attn_modules)

        if not self.enabled:
            return

        root_mod: nn.Module | None = None
        if isinstance(system, nn.Module):
            root_mod = system
        else:
            m = getattr(system, "module", None)
            if isinstance(m, nn.Module):
                root_mod = m

        if root_mod is None:
            self.enabled = False
            return

        # Discover modules and attach stable viz IDs.
        for name, mod in root_mod.named_modules():
            if isinstance(mod, AttentionLayer):
                idx = int(len(self.attn_modules))
                mod._viz_index = int(idx)  # type: ignore[attr-defined]
                mod._viz_name = str(name)  # type: ignore[attr-defined]
                self.attn_modules.append((idx, str(name), mod))
            if isinstance(mod, MemoryBlockLayer):
                idx = int(len(self.mosaic_modules))
                mod._mosaic_index = int(idx)  # type: ignore[attr-defined]
                mod._mosaic_name = str(name)  # type: ignore[attr-defined]
                self.mosaic_modules.append((idx, str(name), mod))

        if not self.attn_modules:
            self.enabled = False
            return

        def _make_hook(i: int):
            def _hook(_m: nn.Module, _inp: tuple[object, ...], out: object) -> None:
                if not self._collector.enabled:
                    return
                y = (
                    out[0]
                    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], Tensor)
                    else out
                )
                if not isinstance(y, Tensor):
                    return
                try:
                    self._collector.observe(i, y)
                except Exception as e:
                    logger.warning(
                        "Failed to observe layer stats (continuing): "
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )

            return _hook

        for idx, name, mod in self.attn_modules:
            try:
                self._handles.append(mod.register_forward_hook(_make_hook(idx)))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to register forward hook on attention layer {name!r} (index {idx}): {e}"
                ) from e

    def begin_step(self, step_1based: int) -> None:
        if not self.enabled:
            return
        self._collector.reset()
        self._collector.enabled = bool((step_1based % self.interval) == 0)

    def end_step(self) -> None:
        if not self.enabled:
            return
        self._collector.enabled = False

    def should_log(self, step_1based: int) -> bool:
        return bool(self.enabled and (step_1based % self.interval) == 0)

    def payload(self) -> dict[str, object]:
        return {"layers": self._collector.finalize()}

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception as e:
                raise RuntimeError("Failed to remove hook") from e
        self._handles.clear()

