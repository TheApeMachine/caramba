"""CLI wrapper for paper artifact collection (legacy convenience).

Preferred usage is manifest-driven via the `paper_collect_artifacts` process target.
This file remains as a convenience wrapper / emergency escape hatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from caramba.config.manifest import Manifest
from caramba.experiment.paper_artifacts import collect_ablation_artifacts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        required=True,
        help="Manifest name (directory under artifacts/). This must match `manifest.name`.",
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        default=["control", "with_null", "tied_qk", "rope_on_sem"],
        help="Targets to include (first is treated as control baseline).",
    )
    ap.add_argument("--artifact-root", default="artifacts", help="Artifact root directory.")
    ap.add_argument("--out-dir", default="artifacts/paper", help="Where to write paper-ready outputs.")
    ap.add_argument(
        "--title",
        default="DBA Ablations (local suite)",
        help="LaTeX table caption / plot title prefix.",
    )
    args = ap.parse_args()

    root = Path(args.artifact_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal manifest stub: we only need `name` for directory resolution.
    # Keep this wrapper to avoid duplicating collection logic.
    m = Manifest.model_validate(
        {
            "version": 2,
            "name": str(args.manifest),
            "notes": "",
            "defaults": {"data": {"tokenizer": "llama", "val_frac": 0.0}, "logging": {"wandb": False, "wandb_project": "", "wandb_entity": ""}, "runtime": {"save_every": 0}},
            "targets": [],
        }
    )
    written = collect_ablation_artifacts(
        manifest=m,
        manifest_path=None,
        artifact_root=root,
        out_dir=out_dir,
        title=str(args.title),
        targets=[str(x) for x in args.targets],
    )
    # Return success if we produced a non-empty JSON with rows.
    try:
        payload = json.loads((out_dir / "ablation_results.json").read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        return 0 if rows else 2
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

