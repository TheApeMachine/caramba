"""Command-line interface for the caramba system.

The CLI is intentionally minimal: one entrypoint that runs a manifest.
"""
from __future__ import annotations

import os
import click
from pathlib import Path
import uvicorn

from caramba_api import app
from caramba.console import logger

from caramba.experiment.runner import run_from_manifest_path
from caramba.codegraph.parser import parse_repo
from caramba.codegraph.sync import sync_files_to_falkordb


class CarambaCLI(click.Group):
    """A click Group that treats unknown commands as `run <manifest_path>`."""
    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            # If the first token isn't a known subcommand, interpret it as:
            # `run <token> ...` so users can do: `caramba path/to/manifest.yml`.
            if args:
                cmd = super().get_command(ctx, "run")
                if cmd is not None:
                    return "run", cmd, args
            raise e


@click.group(cls=CarambaCLI, context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--target",
    type=str,
    default=None,
    help="Target to run (e.g. 'target:<name>' or a bare target name).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate/plan the manifest without executing.",
)
@click.pass_context
def cli(ctx: click.Context, target: str | None, dry_run: bool) -> None:
    """Run a manifest target, or use a subcommand."""
    ctx.ensure_object(dict)
    ctx.obj["target"] = target
    ctx.obj["dry_run"] = bool(dry_run)
    return


@cli.command("run")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--target",
    type=str,
    default=None,
    help="Target to run (e.g. 'target:<name>' or a bare target name). "
    "If omitted, uses entrypoints.default or the first runnable target.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate/plan the manifest without executing.",
)
@click.pass_context
def run(ctx: click.Context, manifest_path: Path, target: str | None, dry_run: bool) -> None:
    """Run what a manifest declares (targets)."""
    # Allow `caramba --dry-run MANIFEST.yml` by inheriting group options.
    if ctx.parent is not None and ctx.parent.obj:
        if target is None:
            target = ctx.parent.obj.get("target")
        if not dry_run:
            dry_run = bool(ctx.parent.obj.get("dry_run", False))
    try:
        result = run_from_manifest_path(manifest_path, target=target, dry_run=dry_run)
        if dry_run and result:
            logger.inspect(result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.Abort()


@cli.command("serve")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8765, show_default=True)
def serve_cmd(host: str, port: int) -> None:
    """Start the local dev API server (UI control-plane).

    This is a lightweight bridge for the frontend demos:
    - POST /api/runs to spawn `caramba run ...`
    - GET  /api/runs/<id>/events to stream `train.jsonl` as SSE
    """
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command("codegraph-sync")
@click.argument("repo_root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=".")
@click.option("--graph", type=str, default="caramba_code", show_default=True, help="FalkorDB graph name.")
@click.option(
    "--falkordb-uri",
    type=str,
    default=None,
    help="FalkorDB URI (e.g. redis://localhost:6379). Defaults to env FALKORDB_URI.",
)
@click.option("--falkordb-host", type=str, default=None, help="FalkorDB host (fallback if no URI).")
@click.option("--falkordb-port", type=int, default=None, help="FalkorDB port (fallback if no URI).")
@click.option("--falkordb-password", type=str, default=None, help="FalkorDB password.")
@click.option(
    "--file",
    "files",
    multiple=True,
    type=str,
    help="Relative file path(s) to sync (repeatable). If omitted, scans all *.py under repo_root.",
)
@click.option("--reset", is_flag=True, default=False, help="Wipe the entire graph before ingesting.")
@click.option(
    "--best-effort/--strict",
    default=True,
    show_default=True,
    help="In best-effort mode, failures won't break hooks/CI.",
)
def codegraph_sync_cmd(
    repo_root: Path,
    graph: str,
    falkordb_uri: str | None,
    falkordb_host: str | None,
    falkordb_port: int | None,
    falkordb_password: str | None,
    files: tuple[str, ...],
    reset: bool,
    best_effort: bool,
) -> None:
    """Parse Python code and sync a structural graph into FalkorDB."""
    # Allow users to disable this (useful for hooks).
    try:
        file_list = [str(x) for x in files if str(x).strip()] or None
        nodes, edges = parse_repo(str(repo_root), files=file_list)
        result = sync_files_to_falkordb(
            repo_root=str(repo_root),
            nodes=nodes,
            edges=edges,
            files=file_list,
            graph=str(graph),
            uri=falkordb_uri,
            host=falkordb_host,
            port=falkordb_port,
            password=falkordb_password,
            reset=bool(reset),
            best_effort=bool(best_effort),
        )

        if result.get("ok"):
            logger.success(
                f"codegraph-sync ok • graph={result.get('graph')} nodes={result.get('nodes')} edges={result.get('edges')}"
            )
        else:
            logger.error(f"codegraph-sync failed: {result.get('error')}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.Abort()


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Returns exit code (0 for success, non-zero for failure).
    """
    try:
        # click's main() expects sys.argv[1:] if args is None
        # standalone_mode=False prevents it from calling sys.exit()
        cli.main(args=argv, standalone_mode=False)
        return 0
    except click.Abort:
        return 1
    except click.ClickException as e:
        e.show()
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
