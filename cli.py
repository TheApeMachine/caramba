"""Command-line interface for the caramba system.

The CLI is intentionally minimal: one entrypoint that runs a manifest.
"""

from __future__ import annotations

import click
from pathlib import Path

from console import logger

from experiment.runner import run_from_manifest_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--target",
    type=str,
    default=None,
    help="Target to run (e.g. 'group:<name>' or 'process:<name>'). "
         "If omitted, uses entrypoints.default or the first runnable target.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate/plan the manifest without executing.",
)
def cli(manifest_path: Path, target: str | None, dry_run: bool) -> None:
    """Run what a manifest declares (training groups or agent processes)."""
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
