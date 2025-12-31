from __future__ import annotations

from caramba.optimizer.kernels import _metal_disabled, metal_kernels_disabled


def test_metal_kernels_disabled_context_disables_all() -> None:
    before = _metal_disabled("rmsnorm")
    with metal_kernels_disabled():
        assert _metal_disabled("rmsnorm") is True
        assert _metal_disabled("rope") is True
    assert _metal_disabled("rmsnorm") == before


def test_metal_kernels_disabled_context_disables_specific_kinds() -> None:
    with metal_kernels_disabled(kinds=["rope"]):
        assert _metal_disabled("rope") is True

