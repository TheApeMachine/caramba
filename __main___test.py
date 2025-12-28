from __future__ import annotations

import importlib
import runpy
import sys

import pytest


def test_caramba_module_main_exits_with_cli_exit_code(monkeypatch) -> None:
    m = importlib.import_module("caramba.__main__")

    def fake_cli_main(_argv=None) -> int:
        return 7

    monkeypatch.setattr(m, "cli_main", fake_cli_main)
    with pytest.raises(SystemExit) as e:
        m.main(["--help"])
    assert e.value.code == 7


def test_caramba_python_m_executes_dunder_main(monkeypatch) -> None:
    # Covers the `if __name__ == "__main__": main()` branch.
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "main", lambda _argv=None: 5)

    # Ensure sys.exit propagates a SystemExit we can assert on.
    monkeypatch.setattr(
        sys,
        "exit",
        lambda code=0: (_ for _ in ()).throw(SystemExit(code)),
    )
    # Avoid runpy warning about pre-imported module.
    sys.modules.pop("caramba.__main__", None)
    with pytest.raises(SystemExit) as e:
        runpy.run_module("caramba.__main__", run_name="__main__")
    assert e.value.code == 5

