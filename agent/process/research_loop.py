"""Autonomous research loop implemented as an agent process.

This is the replacement for the legacy `paper/` package. The loop coordinates:
  code_graph_sync (optional) → paper_write → paper_review → structural_audit → repeat
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from caramba.agent.process import Process
from caramba.agent.process.utils import _extract_json, _manifest_root_dir
from caramba.console import logger
from caramba.config.manifest import Manifest

from caramba.agent.process.paper_review import PaperReview
from caramba.agent.process.paper_write import PaperWrite

if TYPE_CHECKING:
    from caramba.agent import Researcher


@dataclass(frozen=True, slots=True)
class AuditFinding:
    target_layer: str
    change_kind: str
    graph_query_used: str
    dependents_found: list[str]
    audit_passed: bool


class ResearchLoopProcess(Process):
    """Write → review → structural-audit loop."""

    def __init__(
        self,
        agents: dict[str, Researcher],
        *,
        leader_key: str,
        writer_key: str,
        reviewer_key: str,
        max_iterations: int = 5,
        auto_run_experiments: bool = False,
        output_dir: str = "paper",
    ) -> None:
        super().__init__(agents)
        self.leader_key = str(leader_key)
        self.writer_key = str(writer_key)
        self.reviewer_key = str(reviewer_key)
        self.max_iterations = int(max(1, max_iterations))
        self.auto_run_experiments = bool(auto_run_experiments)
        self.output_dir = str(output_dir or "paper")

    async def _structural_audit(
        self,
        *,
        manifest: Manifest,
        manifest_path: Path | None,
        proposed_experiments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask the reviewer agent to run Graphiti queries and return audit JSON."""
        reviewer = self.agents[self.reviewer_key]
        manifest_name = str(getattr(manifest, "name", "") or (manifest_path.stem if manifest_path else "manifest"))

        prompt = (
            "You are performing a structural audit gate.\n\n"
            "For each proposed experiment that changes topology/geometry, you MUST:\n"
            "  1) Use Graphiti tools to query downstream dependents for the target layer.\n"
            "  2) Return STRICT JSON summarizing whether it is safe.\n\n"
            "Use this Cypher query pattern (substitute $target_layer):\n"
            "MATCH (n:Layer {name: $target_layer})<-[:DEPENDS_ON]-(dependent) RETURN dependent\n\n"
            "Return STRICT JSON with keys:\n"
            '  - "manifest": string\n'
            '  - "findings": object[] where each item has:\n'
            '       - "target_layer": string\n'
            '       - "change_kind": string\n'
            '       - "graph_query_used": string\n'
            '       - "dependents_found": string[]\n'
            '       - "audit_passed": boolean\n'
            '  - "overall_passed": boolean\n\n'
            f"<manifest>{manifest_name}</manifest>\n\n"
            "<proposed_experiments_json>\n"
            f"{json.dumps(proposed_experiments, ensure_ascii=False, indent=2)}\n"
            "</proposed_experiments_json>\n"
        )

        res = await reviewer.run(prompt, context=None)
        raw = str(getattr(res, "content", "") or "")
        obj = _extract_json(raw)
        if obj is None:
            return {"overall_passed": False, "error": "unparsed_audit_output", "raw": raw}

        # Normalize findings into the typed AuditFinding shape (and back to JSON-safe dicts).
        raw_findings = obj.get("findings", [])
        typed: list[AuditFinding] = []
        if isinstance(raw_findings, list):
            for f in raw_findings:
                if not isinstance(f, dict):
                    continue
                typed.append(
                    AuditFinding(
                        target_layer=str(f.get("target_layer", "") or ""),
                        change_kind=str(f.get("change_kind", "") or ""),
                        graph_query_used=str(f.get("graph_query_used", "") or ""),
                        dependents_found=[str(x) for x in (f.get("dependents_found", []) or []) if isinstance(x, (str, int, float))],
                        audit_passed=bool(f.get("audit_passed", False)),
                    )
                )
        obj["findings"] = [asdict(x) for x in typed]
        return obj

    async def run(self, *, manifest: Manifest, manifest_path: Path | None) -> dict[str, Any]:
        root = _manifest_root_dir(manifest=manifest, manifest_path=manifest_path)
        loop_dir = root / "agents" / "research_loop"
        loop_dir.mkdir(parents=True, exist_ok=True)

        writer_goal = ""
        last_review: dict[str, Any] | None = None
        iterations: list[dict[str, Any]] = []

        writer_proc = PaperWrite(self.agents, writer_key=self.writer_key, output_dir=self.output_dir)
        reviewer_proc = PaperReview(
            self.agents,
            reviewer_key=self.reviewer_key,
            strictness="conference",
            max_proposed_experiments=3,
            output_dir=self.output_dir,
        )

        for it in range(1, self.max_iterations + 1):
            logger.header("Research Loop", f"iteration {it}/{self.max_iterations}")
            if last_review is not None:
                weaknesses = last_review.get("payload", {}).get("weaknesses", [])
                if isinstance(weaknesses, list) and weaknesses:
                    writer_goal = "Address the following weaknesses:\n- " + "\n- ".join(
                        str(x) for x in weaknesses[:8]
                    )

            w = await writer_proc.run(manifest=manifest, manifest_path=manifest_path, goal=writer_goal)
            r = await reviewer_proc.run(manifest=manifest, manifest_path=manifest_path, goal="")

            last_review = r
            payload = r.get("payload", {})
            proposed = payload.get("proposed_experiments", [])
            if not isinstance(proposed, list):
                proposed = []

            audit = await self._structural_audit(
                manifest=manifest, manifest_path=manifest_path, proposed_experiments=proposed
            )

            iteration_record = {
                "iteration": it,
                "paper_write": w,
                "paper_review": r,
                "structural_audit": audit,
            }
            iterations.append(iteration_record)

            # Persist iteration record.
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            (loop_dir / f"iteration_{it}_{ts}.json").write_text(
                json.dumps(iteration_record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            # Approval heuristics (simple and explicit).
            overall_score = payload.get("overall_score", None)
            recommendation = str(payload.get("recommendation", "") or "").strip().lower()
            style_only = bool(payload.get("style_fixes_only", False))
            audit_passed = bool(audit.get("overall_passed", False))

            if audit_passed and (recommendation == "approve" or (isinstance(overall_score, (int, float)) and float(overall_score) >= 9.0)):
                return {
                    "ok": True,
                    "status": "approved",
                    "final_score": float(overall_score) if isinstance(overall_score, (int, float)) else None,
                    "iterations": iterations,
                }

            if audit_passed and style_only and isinstance(overall_score, (int, float)) and float(overall_score) >= 7.5:
                return {
                    "ok": True,
                    "status": "approved_with_style_fixes",
                    "final_score": float(overall_score),
                    "iterations": iterations,
                }

            # If audit fails, do not proceed to experiment generation.
            if not audit_passed:
                logger.warning("Structural audit failed; rejecting proposed experiments for this iteration.")

            # Experiment generation/execution is intentionally deferred until after audit is reliable
            # (and code_graph_sync has populated the dependency graph).
            if self.auto_run_experiments and audit_passed:
                logger.warning(
                    "auto_run_experiments is enabled, but experiment generation/execution is not yet implemented."
                )

        return {"ok": True, "status": "max_iterations", "iterations": iterations}

