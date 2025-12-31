"""Paper review process driven by the Reviewer persona.

This replaces the legacy `paper/` review workflow by implementing review as a
manifest-driven agent *process* target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from caramba.agent.process import Process
from caramba.agent.process.utils import _extract_json, _manifest_root_dir
from caramba.console import logger
from caramba.config.manifest import Manifest

if TYPE_CHECKING:
    from caramba.agent import Researcher


MAX_PREVIEW_CHARS = 16_000


@dataclass(frozen=True, slots=True)
class PaperReviewResult:
    parsed: bool
    payload: dict[str, Any]


class PaperReview(Process):
    """Review an existing LaTeX paper and propose next actions."""

    def __init__(
        self,
        agents: dict[str, "Researcher"],
        *,
        reviewer_key: str,
        strictness: str = "conference",
        max_proposed_experiments: int = 3,
        output_dir: str = "paper",
    ) -> None:
        super().__init__(agents)
        self.reviewer_key = str(reviewer_key)
        self.strictness = str(strictness or "conference")
        self.max_proposed_experiments = int(max(0, max_proposed_experiments))
        self.output_dir = str(output_dir or "paper")

    async def run(
        self,
        *,
        manifest: Manifest,
        manifest_path: Path | None,
        goal: str = "",
    ) -> dict[str, Any]:
        root = _manifest_root_dir(manifest=manifest, manifest_path=manifest_path)
        paper_tex = root / self.output_dir / "paper.tex"
        if not paper_tex.exists():
            raise ValueError(f"paper_review: missing paper at {paper_tex}")

        text = paper_tex.read_text(encoding="utf-8")
        preview = (
            text if len(text) <= MAX_PREVIEW_CHARS else text[:MAX_PREVIEW_CHARS] + "\n% ... truncated ...\n"
        )

        manifest_notes = str(getattr(manifest, "notes", "") or "")
        manifest_name = str(getattr(manifest, "name", "") or (manifest_path.stem if manifest_path else "manifest"))

        prompt = (
            "You are the Reviewer persona.\n\n"
            "Review the following LaTeX paper draft.\n"
            "You MUST return STRICT JSON.\n\n"
            "Return a JSON object with keys:\n"
            '  - "overall_score": number (0-10)\n'
            '  - "recommendation": string (approve|minor_revisions|major_revisions|reject)\n'
            '  - "summary": string\n'
            '  - "strengths": string[]\n'
            '  - "weaknesses": string[]\n'
            '  - "style_fixes_only": boolean\n'
            '  - "proposed_experiments": object[] (max N) where each has:\n'
            '       - "name": string\n'
            '       - "rationale": string\n'
            '       - "change_kind": string (topology_change|geometry_change|training_change|benchmark_change|docs_only)\n'
            '       - "target_layer": string|null\n\n'
            f"<strictness>{self.strictness}</strictness>\n"
            f"<max_proposed_experiments>{self.max_proposed_experiments}</max_proposed_experiments>\n"
            f"<manifest>\nname: {manifest_name}\nnotes: {manifest_notes}\n</manifest>\n\n"
        )
        if goal.strip():
            prompt += f"<goal>\n{goal.strip()}\n</goal>\n\n"
        prompt += f"<paper_tex>\n{preview}\n</paper_tex>\n\n"

        reviewer = self.agents[self.reviewer_key]
        result = await reviewer.run(prompt, context=None)
        raw = str(getattr(result, "content", "") or "")
        obj = _extract_json(raw)
        parsed = obj is not None
        if obj is None:
            obj = {
                "overall_score": 0.0,
                "recommendation": "reject",
                "summary": raw,
                "strengths": [],
                "weaknesses": ["unparsed_model_output"],
                "style_fixes_only": False,
                "proposed_experiments": [],
            }

        pr = PaperReviewResult(parsed=bool(parsed), payload=obj)
        logger.success(f"Reviewed paper draft: {paper_tex}")

        return {
            "ok": True,
            "process": "paper_review",
            "paper_tex": str(paper_tex),
            "parsed_json": bool(pr.parsed),
            "payload": pr.payload,
        }

