"""Paper writing process driven by the Writer persona.

This replaces the legacy `paper/` package workflow by implementing paper drafting
as a manifest-driven agent *process* target.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from caramba.agent.process import Process
from caramba.agent.process.utils import _extract_json, _manifest_root_dir
from caramba.console import logger
from caramba.config.manifest import Manifest

if TYPE_CHECKING:
    from caramba.agent import Researcher


@dataclass(frozen=True, slots=True)
class PaperWriteResult:
    paper_tex_path: Path
    version_backup_path: Path | None
    parsed: bool
    payload: dict[str, Any]


class PaperWrite(Process):
    """Write or update a LaTeX paper artifact."""

    def __init__(
        self,
        agents: dict[str, Researcher],
        *,
        writer_key: str,
        output_dir: str = "paper",
    ) -> None:
        super().__init__(agents)
        self.writer_key = str(writer_key)
        self.output_dir = str(output_dir or "paper")

    async def run(
        self,
        *,
        manifest: Manifest,
        manifest_path: Path | None,
        goal: str = "",
    ) -> dict[str, Any]:
        root = _manifest_root_dir(manifest=manifest, manifest_path=manifest_path)
        out_dir = root / self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        paper_tex = out_dir / "paper.tex"
        versions = out_dir / "versions"
        versions.mkdir(parents=True, exist_ok=True)

        prev: str = ""
        backup: Path | None = None
        if paper_tex.exists():
            try:
                prev = paper_tex.read_text(encoding="utf-8")
            except OSError:
                prev = ""
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup = versions / f"paper_{ts}.tex"
            try:
                backup.write_text(prev, encoding="utf-8")
            except OSError:
                backup = None

        writer = self.agents[self.writer_key]
        manifest_notes = str(getattr(manifest, "notes", "") or "")
        manifest_name = str(getattr(manifest, "name", "") or (manifest_path.stem if manifest_path else "manifest"))

        prompt = (
            "You are the Writer persona.\n\n"
            "Write or update a LaTeX research paper draft.\n\n"
            "Return STRICT JSON with keys:\n"
            '  - "title": string\n'
            '  - "latex": string (the full LaTeX document)\n'
            '  - "notes": string (optional)\n\n'
            f"<manifest>\nname: {manifest_name}\nnotes: {manifest_notes}\n</manifest>\n\n"
        )
        if goal.strip():
            prompt += f"<goal>\n{goal.strip()}\n</goal>\n\n"
        if prev.strip():
            # Keep context bounded.
            preview = prev if len(prev) <= 12_000 else prev[:12_000] + "\n% ... truncated ...\n"
            prompt += f"<current_paper>\n{preview}\n</current_paper>\n\n"
            prompt += "Update the existing paper; keep it coherent.\n\n"
        else:
            prompt += (
                "No existing paper found. Create a complete LaTeX document.\n"
                "Use standard packages (amsmath, amssymb, graphicx, hyperref, natbib).\n\n"
            )

        result = await writer.run(prompt, context=None)
        raw = str(getattr(result, "content", "") or "")
        obj = _extract_json(raw)
        parsed = obj is not None
        if obj is None:
            obj = {"title": manifest_name, "latex": raw, "notes": "unparsed_model_output"}

        latex = str(obj.get("latex", "") or "")
        if not latex.strip():
            latex = raw
            obj["latex"] = latex
            obj["notes"] = str(obj.get("notes", "") or "") + " missing_latex_field"

        paper_tex.write_text(latex, encoding="utf-8")
        logger.success(f"Wrote paper draft: {paper_tex}")

        pw = PaperWriteResult(
            paper_tex_path=paper_tex,
            version_backup_path=backup,
            parsed=bool(parsed),
            payload=obj,
        )

        return {
            "ok": True,
            "process": "paper_write",
            "paper_tex": str(pw.paper_tex_path),
            "backup": str(pw.version_backup_path) if pw.version_backup_path else None,
            "parsed_json": bool(pw.parsed),
            "payload": pw.payload,
        }

