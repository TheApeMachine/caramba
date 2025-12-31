"""Platform improvement pipeline (manifest-driven).

This is a process-only workflow that:
  - syncs the current repo + model topology into Graphiti/FalkorDB (via MCP)
  - asks multiple agents to propose improvements
  - runs a consensus discussion to pick ONE improvement + plan
  - has a developer agent generate a git patch
  - applies the patch on a new branch, runs tests, and gates with a reviewer
  - opens a PR (via `gh`) when accepted
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from caramba.agent.process import Process
from caramba.agent.process.discussion import Discussion
from caramba.agent.process.utils import _extract_json, _manifest_root_dir
from caramba.console import logger
from caramba.config.manifest import Manifest

from caramba.agent.process.code_graph_sync import CodeGraphSync

if TYPE_CHECKING:
    from caramba.agent import Researcher


_log = logging.getLogger(__name__)


def _run(cmd: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return str(proc.stdout or "")


def _git_root(repo_root: Path) -> Path:
    out = _run(["git", "rev-parse", "--show-toplevel"], cwd=repo_root).strip()
    return Path(out)


def _git_current_branch(repo_root: Path) -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).strip()


def _git_ls_files(repo_root: Path) -> list[str]:
    out = _run(["git", "ls-files"], cwd=repo_root)
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:48] if len(s) > 48 else s


def _extract_patch(text: str) -> str | None:
    """Best-effort extraction of a unified diff patch from model output."""
    s = str(text or "")
    # Prefer fenced ```diff blocks.
    m = re.search(r"```diff\s+([\s\S]*?)```", s)
    if m:
        return m.group(1).strip() + "\n"
    # Otherwise, locate a `diff --git` block.
    m2 = re.search(r"(diff --git[\s\S]*)", s)
    if m2:
        return m2.group(1).strip() + "\n"
    return None


@dataclass(frozen=True, slots=True)
class Idea:
    title: str
    rationale: str
    scope: str
    risks: list[str]
    files: list[str]
    test_plan: list[str]


class PlatformImprove(Process):
    """End-to-end platform improvement pipeline."""

    def __init__(
        self,
        agents: dict[str, "Researcher"],
        *,
        # ingestion
        ingest_agent: str,
        index_namespace: str,
        ingest_repo: bool,
        ingest_models: bool,
        max_files: int,
        max_chars_per_file: int,
        # roles
        leader_key: str,
        ideator_keys: list[str],
        developer_key: str,
        reviewer_key: str,
        # dev automation
        repo_root: str,
        base_branch: str,
        branch_prefix: str,
        tests: list[str],
        max_review_rounds: int,
        open_pr: bool,
        pr_title_prefix: str,
        topic: str,
    ) -> None:
        super().__init__(agents)
        self.ingest_agent = str(ingest_agent)
        self.index_namespace = str(index_namespace or "main")
        self.ingest_repo = bool(ingest_repo)
        self.ingest_models = bool(ingest_models)
        self.max_files = int(max(1, max_files))
        self.max_chars_per_file = int(max(256, max_chars_per_file))

        self.leader_key = str(leader_key)
        self.ideator_keys = [str(x) for x in (ideator_keys or [])]
        self.developer_key = str(developer_key)
        self.reviewer_key = str(reviewer_key)

        self.repo_root = str(repo_root or ".")
        self.base_branch = str(base_branch or "main")
        self.branch_prefix = str(branch_prefix or "agent/platform-improve")
        self.tests = [str(x) for x in (tests or [])]
        self.max_review_rounds = int(max(0, max_review_rounds))
        self.open_pr = bool(open_pr)
        self.pr_title_prefix = str(pr_title_prefix or "")
        self.topic = str(topic or "").strip() or "Propose and implement one high-impact improvement."

    async def _ingest_repo_to_graphiti(self, *, manifest: Manifest, manifest_path: Path | None) -> dict[str, Any]:
        """Summarize git-tracked files and ingest into Graphiti via MCP."""
        repo_root = _git_root(Path(self.repo_root))
        files = _git_ls_files(repo_root)[: self.max_files]

        records: list[dict[str, Any]] = []
        for rel in files:
            p = repo_root / rel
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                txt = ""
            preview = txt[: self.max_chars_per_file]
            records.append(
                {
                    "path": rel,
                    "bytes": int(p.stat().st_size) if p.exists() else 0,
                    "preview": preview,
                }
            )

        manifest_name = str(getattr(manifest, "name", "") or (manifest_path.stem if manifest_path else "manifest"))
        branch = _git_current_branch(repo_root)
        payload = {
            "kind": "caramba_repo_snapshot",
            "manifest": manifest_name,
            "repo_root": str(repo_root),
            "branch": branch,
            "max_files": self.max_files,
            "max_chars_per_file": self.max_chars_per_file,
            "files": records,
        }

        agent = self.agents[self.ingest_agent]
        prompt = (
            "You are performing deterministic graph ingestion for a repository snapshot.\n\n"
            "You MUST call the Graphiti tool `add_memory` exactly once with:\n"
            '  - name: "caramba_repo_snapshot"\n'
            "  - episode_body: the JSON payload below\n"
            '  - source: "json"\n'
            f'  - group_id: "{self.index_namespace}"\n\n'
            'Return STRICT JSON: {"ok": true}.\n\n'
            "<payload_json>\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
            "</payload_json>\n"
        )
        res = await agent.run(prompt, context=None)
        raw = str(getattr(res, "content", "") or "")
        obj = _extract_json(raw) or {}
        ok = bool(obj.get("ok", False))
        if not ok:
            logger.warning("repo ingestion: agent did not return ok=true; continuing")
        return {"ok": bool(ok), "files_ingested": len(records), "agent_response": raw}

    async def _ideate(self, *, manifest: Manifest, manifest_path: Path | None) -> list[Idea]:
        manifest_name = str(getattr(manifest, "name", "") or (manifest_path.stem if manifest_path else "manifest"))
        ideas: list[Idea] = []
        for k in self.ideator_keys:
            if k not in self.agents:
                continue
            a = self.agents[k]
            prompt = (
                "You are an agent ideating improvements to the Caramba platform.\n\n"
                "You MAY query Graphiti (FalkorDB) and DeepLake via your tools if available.\n"
                "Your job: propose 2-3 concrete improvements that are implementable in one PR.\n\n"
                "Return STRICT JSON with:\n"
                '  - "ideas": [{\n'
                '      "title": string,\n'
                '      "rationale": string,\n'
                '      "scope": string,\n'
                '      "risks": string[],\n'
                '      "files": string[],\n'
                '      "test_plan": string[]\n'
                "  }]\n\n"
                f"<manifest>{manifest_name}</manifest>\n"
                f"<topic>{self.topic}</topic>\n"
            )
            res = await a.run(prompt, context=None)
            raw = str(getattr(res, "content", "") or "")
            obj = _extract_json(raw)
            if not obj or not isinstance(obj.get("ideas", None), list):
                continue
            for it in obj["ideas"]:
                if not isinstance(it, dict):
                    continue
                ideas.append(
                    Idea(
                        title=str(it.get("title", "") or "").strip(),
                        rationale=str(it.get("rationale", "") or "").strip(),
                        scope=str(it.get("scope", "") or "").strip(),
                        risks=[str(x) for x in (it.get("risks", []) or []) if isinstance(x, (str, int, float))],
                        files=[str(x) for x in (it.get("files", []) or []) if isinstance(x, (str, int, float))],
                        test_plan=[str(x) for x in (it.get("test_plan", []) or []) if isinstance(x, (str, int, float))],
                    )
                )
        # Drop empty titles.
        return [x for x in ideas if x.title]

    async def _consensus_plan(self, *, ideas: list[Idea]) -> dict[str, Any]:
        # Use a real multi-agent discussion so they converge.
        leader = self.agents[self.leader_key]
        team: dict[str, Researcher] = {self.leader_key: leader}
        for k in self.ideator_keys:
            if k in self.agents:
                team[k] = self.agents[k]

        discussion = Discussion(agents=team, team_leader_key=self.leader_key, prompts_dir=Path("config/prompts"))
        ideas_json = json.dumps([asdict(x) for x in ideas], ensure_ascii=False, indent=2)
        topic = (
            f"{self.topic}\n\n"
            "Use the following candidate ideas and converge on ONE improvement.\n"
            "The outcome MUST be a STRICT JSON plan (no prose) including:\n"
            '  - "selection": { "title": string, "why": string }\n'
            '  - "branch_slug": string\n'
            '  - "implementation_plan": string[] (ordered steps)\n'
            '  - "files_touched": string[]\n'
            '  - "tests": string[]\n'
            '  - "pr": { "title": string, "body": string }\n\n'
            "<candidate_ideas_json>\n"
            f"{ideas_json}\n"
            "</candidate_ideas_json>\n"
        )
        res = await discussion.run(topic, context=None)
        plan = _extract_json(str(res.get("conclusion", "") or ""))
        return plan or {"error": "unparsed_plan", "raw": res}

    async def _developer_patch(self, *, plan: dict[str, Any], diff_context: str) -> str:
        dev = self.agents[self.developer_key]
        prompt = (
            "You are the Developer agent.\n\n"
            "Implement the approved plan by producing a SINGLE unified git patch.\n"
            "Constraints:\n"
            "  - Output ONLY a patch (prefer a ```diff fenced block).\n"
            "  - Do not include explanations.\n"
            "  - Keep changes scoped to the plan.\n\n"
            "<plan_json>\n"
            f"{json.dumps(plan, ensure_ascii=False, indent=2)}\n"
            "</plan_json>\n\n"
            "<repo_diff_context>\n"
            f"{diff_context}\n"
            "</repo_diff_context>\n"
        )
        res = await dev.run(prompt, context=None)
        raw = str(getattr(res, "content", "") or "")
        patch = _extract_patch(raw)
        if patch is None:
            raise ValueError("Developer did not return a parseable git patch.")
        return patch

    async def _review(self, *, plan: dict[str, Any], diff_text: str) -> dict[str, Any]:
        reviewer = self.agents[self.reviewer_key]
        prompt = (
            "You are the Reviewer agent.\n\n"
            "Review the proposed change and either ACCEPT or REQUEST_CHANGES.\n"
            "Return STRICT JSON:\n"
            '  - "decision": "accept"|"request_changes"\n'
            '  - "summary": string\n'
            '  - "required_changes": string[]\n'
            '  - "risks": string[]\n'
            '  - "test_plan": string[]\n\n'
            "<plan_json>\n"
            f"{json.dumps(plan, ensure_ascii=False, indent=2)}\n"
            "</plan_json>\n\n"
            "<git_diff>\n"
            f"{diff_text}\n"
            "</git_diff>\n"
        )
        res = await reviewer.run(prompt, context=None)
        raw = str(getattr(res, "content", "") or "")
        obj = _extract_json(raw) or {"decision": "request_changes", "summary": raw, "required_changes": ["unparsed_output"], "risks": [], "test_plan": []}
        return obj

    def _apply_patch_and_commit(self, *, repo_root: Path, patch: str, commit_message: str) -> None:
        # Apply patch.
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=str(repo_root),
            input=patch,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git apply failed:\n{proc.stderr}\n")
        _ = _run(["git", "add", "-A"], cwd=repo_root)
        _ = _run(["git", "commit", "-m", commit_message], cwd=repo_root)

    def _run_tests(self, *, repo_root: Path) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for cmd in self.tests:
            if not cmd.strip():
                continue
            # shell=True for convenience here; command is controlled by manifest.
            proc = subprocess.run(cmd, cwd=str(repo_root), shell=True, text=True, capture_output=True)
            results.append(
                {
                    "cmd": cmd,
                    "ok": proc.returncode == 0,
                    "returncode": int(proc.returncode),
                    "stdout": proc.stdout[-20000:],
                    "stderr": proc.stderr[-20000:],
                }
            )
            if proc.returncode != 0:
                break
        return results

    def _open_pr(self, *, repo_root: Path, title: str, body: str, branch: str) -> str:
        _ = _run(["git", "push", "-u", "origin", branch], cwd=repo_root)
        # `gh pr create` prints a URL on success.
        out = _run(["gh", "pr", "create", "--title", title, "--body", body], cwd=repo_root)
        return out.strip()

    async def run(self, *, manifest: Manifest, manifest_path: Path | None) -> dict[str, Any]:
        root = _manifest_root_dir(manifest=manifest, manifest_path=manifest_path)
        out_dir = root / "agents" / "platform_improve"
        out_dir.mkdir(parents=True, exist_ok=True)

        repo_root = _git_root(Path(self.repo_root))
        start_branch = _git_current_branch(repo_root)

        ingest: dict[str, Any] = {"ok": True, "steps": []}
        if self.ingest_models:
            # Only works if manifest contains experiment targets with system.config.model.
            try:
                sync = CodeGraphSync(self.agents, agent_key=self.ingest_agent, index_namespace=self.index_namespace)
                ingest["steps"].append(await sync.run(manifest=manifest, manifest_path=manifest_path))
            except Exception as e:
                ingest["steps"].append({"ok": False, "error": f"model_ingest_failed: {e}"})
        if self.ingest_repo:
            try:
                ingest["steps"].append(await self._ingest_repo_to_graphiti(manifest=manifest, manifest_path=manifest_path))
            except Exception as e:
                ingest["steps"].append({"ok": False, "error": f"repo_ingest_failed: {e}"})

        ideas = await self._ideate(manifest=manifest, manifest_path=manifest_path)
        plan = await self._consensus_plan(ideas=ideas)

        # Build branch name.
        sel = plan.get("selection", {}) if isinstance(plan.get("selection", None), dict) else {}
        branch_slug = str(plan.get("branch_slug", "") or _slug(str(sel.get("title", "improvement"))))
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        branch = f"{self.branch_prefix}/{ts}-{branch_slug}"

        # Checkout base, create branch.
        _ = _run(["git", "checkout", self.base_branch], cwd=repo_root)
        _ = _run(["git", "pull", "--ff-only"], cwd=repo_root)
        _ = _run(["git", "checkout", "-b", branch], cwd=repo_root)

        rounds: list[dict[str, Any]] = []
        accepted = False
        pr_url: str | None = None

        try:
            for r in range(max(1, self.max_review_rounds + 1)):
                diff_ctx = _run(["git", "status", "--porcelain=v1"], cwd=repo_root) + "\n" + _run(
                    ["git", "diff"], cwd=repo_root
                )
                patch = await self._developer_patch(plan=plan, diff_context=diff_ctx)
                commit_msg = str(sel.get("title", "") or "platform improvement").strip() or "platform improvement"
                self._apply_patch_and_commit(repo_root=repo_root, patch=patch, commit_message=commit_msg)

                diff_text = _run(["git", "show", "--stat"], cwd=repo_root) + "\n" + _run(["git", "show"], cwd=repo_root)
                test_results = self._run_tests(repo_root=repo_root)
                review = await self._review(plan=plan, diff_text=diff_text)

                rounds.append({"round": r, "tests": test_results, "review": review})
                decision = str(review.get("decision", "") or "").strip().lower()
                if decision == "accept" and all(bool(t.get("ok", False)) for t in test_results):
                    accepted = True
                    break

                # If changes requested, reset back one commit and iterate.
                try:
                    show = _run(["git", "show", "--stat", "--patch", "HEAD"], cwd=repo_root)
                except RuntimeError:
                    show = ""
                    _log.debug("Failed to capture git patch before reset", exc_info=True)
                if show.strip():
                    logger.info("About to reset, current diff/patch (git show HEAD):")
                    try:
                        logger.panel(show[-20000:], title="About to reset: git show HEAD", style="warning")
                    except Exception:
                        # Best-effort: never crash because we failed to render the patch.
                        _log.debug("Failed to render git patch panel", exc_info=True)
                try:
                    ts2 = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    patch_path = out_dir / f"about_to_reset_round_{r}_{ts2}.patch"
                    patch_path.write_text(show, encoding="utf-8")
                    logger.info(f"Saved pre-reset patch to: {patch_path}")
                except Exception:
                    _log.debug("Failed to write pre-reset patch file", exc_info=True)
                _ = _run(["git", "reset", "--hard", "HEAD~1"], cwd=repo_root)

            if accepted and self.open_pr:
                pr = plan.get("pr", {}) if isinstance(plan.get("pr", None), dict) else {}
                title = str(pr.get("title", "") or sel.get("title", "") or "Platform improvement").strip()
                if self.pr_title_prefix and not title.startswith(self.pr_title_prefix):
                    title = f"{self.pr_title_prefix}{title}"
                body = str(pr.get("body", "") or "").strip()
                pr_url = self._open_pr(repo_root=repo_root, title=title, body=body, branch=branch)
        finally:
            # Leave repo on the branch we created (useful), but if anything blew up early,
            # try to restore the original branch.
            try:
                if not accepted and _git_current_branch(repo_root) != start_branch:
                    _ = _run(["git", "checkout", start_branch], cwd=repo_root)
            except Exception as e:
                _log.debug(f"Failed to restore git branch ({start_branch}): {e}", exc_info=True)

        payload = {
            "ok": bool(accepted),
            "process": "platform_improve",
            "topic": self.topic,
            "index_namespace": self.index_namespace,
            "ingest": ingest,
            "ideas": [asdict(x) for x in ideas],
            "plan": plan,
            "branch": branch,
            "rounds": rounds,
            "pr_url": pr_url,
        }
        (out_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

