from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import pytest

from caramba.config.component import ComponentSpec
from caramba.runtime.registry import ComponentRegistry, _construct, _import_symbol


class Dummy:
    def __init__(self, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        # Accept either dict-style config or kwargs; store both for assertions.
        self.config = config
        self.kwargs = dict(kwargs)


class MutatingDummy:
    def __init__(self, a: dict[str, Any]) -> None:
        # Mutate nested config to ensure the registry deep-copies before construction.
        a["b"] = 2
        self.a = a


def dummy_factory(**kwargs: Any) -> Dummy:
    return Dummy(**kwargs)


def test_import_symbol_imports_and_validates_path() -> None:
    sym = _import_symbol("caramba.runtime.registry_test:Dummy")
    # Depending on how pytest imports test modules, this file can be imported
    # under multiple module names, so identity with the local `Dummy` symbol is
    # not stable. Assert on the imported symbol itself instead.
    assert getattr(sym, "__name__", None) == "Dummy"
    assert str(getattr(sym, "__module__", "")).endswith("caramba.runtime.registry_test")

    with pytest.raises(
        ValueError,
        match=r"Invalid python symbol path .*Expected 'module:Symbol'\.",
    ):
        _import_symbol("no_colon_here")

    with pytest.raises(
        ImportError,
        match=r"Symbol 'DoesNotExist' not found in module 'caramba\.runtime\.registry_test'",
    ):
        _import_symbol("caramba.runtime.registry_test:DoesNotExist")


def test_construct_prefers_kwargs_then_falls_back_to_positional() -> None:
    obj = _construct(dummy_factory, {"x": 1})
    assert isinstance(obj, Dummy)
    assert obj.kwargs["x"] == 1

    # For class types, implementation prefers dict-style first, then kwargs.
    obj2 = _construct(Dummy, {"y": 2})
    assert isinstance(obj2, Dummy)


def test_registry_resolve_python_escape_hatch_and_build() -> None:
    r = ComponentRegistry()
    spec = ComponentSpec(ref="x", impl="python:caramba.runtime.registry_test:Dummy", config={"a": 1})
    resolved = r.resolve(spec, backend="torch")
    assert resolved.ref == "x"
    assert getattr(resolved.factory, "__name__", None) == "Dummy"

    built = r.build(spec, backend="torch")
    assert isinstance(built, resolved.factory)  # type: ignore[arg-type]


def test_registry_register_and_resolve_backend_mapping() -> None:
    r = ComponentRegistry()
    r.register(backend="torch", ref="dummy", python="caramba.runtime.registry_test:Dummy")
    spec = ComponentSpec(ref="dummy", config={"z": 3})
    resolved = r.resolve(spec, backend="torch")
    assert resolved.impl == "caramba.runtime.registry_test:Dummy"
    built = r.build(spec, backend="torch")
    assert isinstance(built, resolved.factory)  # type: ignore[arg-type]


def test_registry_build_deepcopies_config_to_prevent_mutation_bleed() -> None:
    r = ComponentRegistry()
    r.register(backend="torch", ref="mut", python="caramba.runtime.registry_test:MutatingDummy")
    nested = {"b": 1}
    spec = ComponentSpec(ref="mut", config={"a": nested})
    r.build(spec, backend="torch")
    assert nested["b"] == 1
    assert spec.config["a"]["b"] == 1

