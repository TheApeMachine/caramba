"""Component registry and python escape hatch.

The manifest references components via stable semantic ids (`ref`). The runtime
registry maps these ids to concrete Python implementations for a given backend.
"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any, Callable

from config.component import ComponentSpec


def _import_symbol(path: str) -> object:
    """Import a symbol from a `module:Symbol` string."""
    if ":" not in path:
        raise ValueError(f"Invalid python symbol path {path!r}. Expected 'module:Symbol'.")
    mod_name, sym = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, sym)
    except AttributeError as e:
        raise ImportError(f"Symbol {sym!r} not found in module {mod_name!r}") from e


def _construct(factory: object, config: dict[str, Any]) -> object:
    """Strict constructor: all components must accept keyword arguments.

    Rationale:
    - Avoid hidden fallbacks that can double-nest configs (e.g. {"model": {...}})
      or silently select an unintended construction path.
    - Keep manifests + components "one way": config keys map directly to init kwargs.
    """
    if not callable(factory):
        raise TypeError(f"Cannot construct component from {type(factory).__name__}")
    try:
        return factory(**config)  # type: ignore[misc]
    except TypeError as e:
        # Provide an actionable error message showing what we tried.
        raise TypeError(
            "Component construction failed. All components must accept keyword arguments "
            "(Component(**config)).\n\n"
            f"Factory: {factory!r}\n"
            f"Config keys: {sorted(map(str, config.keys()))}\n"
            f"Error: {e}"
        ) from e


@dataclass(frozen=True, slots=True)
class ResolvedComponent:
    ref: str
    impl: str
    factory: object


class ComponentRegistry:
    """Resolve `ComponentSpec` into concrete implementations."""

    def __init__(self) -> None:
        # backend -> ref -> pythonPath (module:Symbol)
        self._by_backend: dict[str, dict[str, str]] = {"torch": {}}

    def register(self, *, backend: str, ref: str, python: str) -> None:
        self._by_backend.setdefault(str(backend), {})[str(ref)] = str(python)

    def resolve(self, spec: ComponentSpec, *, backend: str) -> ResolvedComponent:
        if spec.impl and str(spec.impl).startswith("python:"):
            py = str(spec.impl).removeprefix("python:")
            factory = _import_symbol(py)
            return ResolvedComponent(ref=spec.ref, impl=str(spec.impl), factory=factory)

        b = str(spec.impl or backend or "torch").strip() or "torch"
        table = self._by_backend.get(b, {})
        if spec.ref not in table:
            raise KeyError(f"Unknown component ref={spec.ref!r} for backend={b!r}")
        py = table[spec.ref]
        factory = _import_symbol(py)
        return ResolvedComponent(ref=spec.ref, impl=py, factory=factory)

    def build(self, spec: ComponentSpec, *, backend: str) -> object:
        resolved = self.resolve(spec, backend=backend)
        # IMPORTANT: manifest-derived config may contain shared nested objects
        # (e.g., YAML anchors/aliases). Deep-copy to avoid cross-target mutation
        # and keep component construction isolated.
        cfg: dict[str, Any] = copy.deepcopy(spec.config)
        return _construct(resolved.factory, cfg)

