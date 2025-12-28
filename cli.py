"""Command-line interface for the caramba system.

The only two commands are compile and run. You can look at compile
as a dry-run of the experiment, to validate the manifest and print the plan.
"""
from __future__ import annotations

import click
import json
import asyncio
from pathlib import Path

from compiler import Compiler
from compiler.plan import Planner
from config.manifest import Manifest
from console import logger

from experiment.runner import ExperimentRunner

compiler = Compiler()


@click.group()
def cli() -> None:
    """Caramba - A research platform for efficient AI."""
    pass


@cli.command(name="compile")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--print-plan",
    is_flag=True,
    help="Print the lowered graph/plan.",
)
def compile_cmd(manifest_path: Path, print_plan: bool) -> None:
    """Compile a manifest (parse → lower → validate), without building."""
    try:
        compiler.validator.validate_manifest(
            compiler.lowerer.lower_manifest(
                Manifest.from_path(manifest_path)
            ), print_plan=print_plan
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.Abort()


@cli.command(name="run")
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--group",
    type=str,
    default=None,
    help="Group name to run. If not specified, runs the first group.",
)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Resume from a saved checkpoint (.pt).",
)
@click.option(
    "--benchmarks-only",
    is_flag=True,
    default=False,
    help="Skip training runs and only execute benchmarks/artifacts "
         "(typically used with --resume-from).",
)
def run_cmd(
    manifest_path: Path,
    group: str | None,
    resume_from: Path | None,
    benchmarks_only: bool,
) -> None:
    """Run a full experiment: upcycle + benchmarks + artifact generation."""
    try:
        ExperimentRunner(compiler.validator.validate_manifest(
            compiler.lowerer.lower_manifest(
                Manifest.from_path(manifest_path)
            ), print_plan=True
        )).run(group, resume_from=resume_from, benchmarks_only=benchmarks_only)
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
