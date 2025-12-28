"""Command-line interface for the caramba system.

The CLI is intentionally minimal: one entrypoint that runs a manifest.
"""

from __future__ import annotations

import click
from pathlib import Path

from console import logger

from experiment.runner import run_from_manifest_path


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
