from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import click
from click.testing import CliRunner

import cli as cli_mod


def test_cli_run_invokes_manifest_runner(monkeypatch) -> None:
    runner = CliRunner()
    called: dict[str, object] = {}

    def fake_run_from_manifest_path(manifest_path: Path, *, target: str | None = None, dry_run: bool = False):
        called["manifest_path"] = manifest_path
        called["target"] = target
        called["dry_run"] = dry_run
        return {"ok": True}

    monkeypatch.setattr(cli_mod, "run_from_manifest_path", fake_run_from_manifest_path)

    with runner.isolated_filesystem():
        p = Path("m.yml")
        p.write_text("name: test\n", encoding="utf-8")
        res = runner.invoke(cli_mod.cli, ["run", str(p), "--dry-run"])

    assert res.exit_code == 0, res.output
    manifest_path = called["manifest_path"]
    assert isinstance(manifest_path, Path)
    assert manifest_path.name == "m.yml"
    assert called["target"] is None
    assert called["dry_run"] is True


def test_cli_unknown_command_is_treated_as_run(monkeypatch) -> None:
    runner = CliRunner()
    called: dict[str, object] = {}

    def fake_run_from_manifest_path(manifest_path: Path, *, target: str | None = None, dry_run: bool = False):
        called["manifest_path"] = manifest_path
        called["target"] = target
        called["dry_run"] = dry_run
        return {"ok": True}

    monkeypatch.setattr(cli_mod, "run_from_manifest_path", fake_run_from_manifest_path)

    with runner.isolated_filesystem():
        p = Path("m.yml")
        p.write_text("name: test\n", encoding="utf-8")
        # First token is not a known subcommand => CarambaCLI should interpret it as `run <token> ...`.
        res = runner.invoke(cli_mod.cli, [str(p), "--dry-run"])

    assert res.exit_code == 0, res.output
    manifest_path = called["manifest_path"]
    assert isinstance(manifest_path, Path)
    assert manifest_path.name == "m.yml"
    assert called["dry_run"] is True


def test_cli_group_dry_run_option_is_inherited_by_run_command(monkeypatch) -> None:
    runner = CliRunner()
    called: dict[str, object] = {}

    def fake_run_from_manifest_path(manifest_path: Path, *, target: str | None = None, dry_run: bool = False):
        called["manifest_path"] = manifest_path
        called["target"] = target
        called["dry_run"] = dry_run
        return {"ok": True}

    monkeypatch.setattr(cli_mod, "run_from_manifest_path", fake_run_from_manifest_path)

    with runner.isolated_filesystem():
        p = Path("m.yml")
        p.write_text("name: test\n", encoding="utf-8")
        res = runner.invoke(cli_mod.cli, ["--dry-run", "run", str(p)])

    assert res.exit_code == 0, res.output
    assert called["dry_run"] is True


def test_cli_main_returns_zero_on_help() -> None:
    # Uses click's help path; should not raise or exit the interpreter.
    exit_code = cli_mod.main(["--help"])
    assert exit_code == 0


def test_cli_main_returns_one_on_click_abort(monkeypatch) -> None:
    def boom(*_args, **_kwargs) -> NoReturn:
        raise click.Abort()

    monkeypatch.setattr(cli_mod.cli, "main", boom)
    assert cli_mod.main(["--help"]) == 1

