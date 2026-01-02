"""Idle process: turn spare cycles into durable artifacts.

This process is intended to be invoked when the system is "idle" (e.g. user not
actively interacting). It executes a bounded set of safe, measurable tasks:

- Readiness: refresh the knowledge graph index of the current manifest/model.
- Continuous evaluation: run short deterministic commands (tests/bench slices).
- Research loop: run a short paper write → review → audit iteration.

The key property is a strict wall-clock budget: the process must stop once the
budget is exhausted and return a diffable JSON result summarizing what ran.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from caramba.agent.process import Process
from caramba.agent.process.code_graph_sync import CodeGraphSync
from caramba.agent.process.research_loop import ResearchLoopProcess
from caramba.console import logger
from caramba.config.manifest import Manifest

if TYPE_CHECKING:
    from caramba.agent import Researcher


@dataclass(frozen=True, slots=True)
class _CmdResult:
    cmd: str
    cwd: str
    ok: bool
    returncode: int
    duration_sec: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str


def _tail(s: str, *, limit: int = 20_000) -> str:
    s2 = str(s or "")
    return s2[-limit:] if len(s2) > limit else s2


class IdleProcess(Process):
    """Budgeted idle loop process."""

    def __init__(
        self,
        agents: dict[str, "Researcher"],
        *,
        max_wall_time_sec: int = 600,
        # readiness
        run_code_graph_sync: bool = True,
        code_graph_sync_agent: str = "leader",
        index_namespace: str = "main",
        # evaluation
        run_eval: bool = False,
        eval_cmds: list[str] | None = None,
        eval_timeout_sec: int = 300,
        eval_cwd: str = ".",
        # research loop
        run_research_loop: bool = True,
        leader_key: str = "leader",
        writer_key: str = "writer",
        reviewer_key: str = "reviewer",
        research_max_iterations: int = 1,
        research_auto_run_experiments: bool = False,
        output_dir: str = "paper",
    ) -> None:
        super().__init__(agents)
        self.max_wall_time_sec = int(max(1, max_wall_time_sec))

        self.run_code_graph_sync = bool(run_code_graph_sync)
        self.code_graph_sync_agent = str(code_graph_sync_agent)
        self.index_namespace = str(index_namespace or "main")

        self.run_eval = bool(run_eval)
        self.eval_cmds = [str(x) for x in (eval_cmds or []) if str(x).strip()]
        self.eval_timeout_sec = int(max(1, eval_timeout_sec))
        self.eval_cwd = str(eval_cwd or ".")

        self.run_research_loop = bool(run_research_loop)
        self.leader_key = str(leader_key)
        self.writer_key = str(writer_key)
        self.reviewer_key = str(reviewer_key)
        self.research_max_iterations = int(max(1, research_max_iterations))
        self.research_auto_run_experiments = bool(research_auto_run_experiments)
        self.output_dir = str(output_dir or "paper")

    def _time_left(self, *, deadline: float) -> float:
        return float(deadline - time.monotonic())

    def _run_cmd(self, *, cmd: str, cwd: Path, timeout_sec: int) -> _CmdResult:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                shell=True,
                text=True,
                capture_output=True,
                timeout=float(timeout_sec),
            )
            timed_out = False
            rc = int(proc.returncode)
            ok = rc == 0
            out = str(proc.stdout or "")
            err = str(proc.stderr or "")
        except subprocess.TimeoutExpired as e:
            timed_out = True
            ok = False
            rc = 124
            out = str(getattr(e, "stdout", "") or "")
            err = str(getattr(e, "stderr", "") or "")
        dt = float(time.perf_counter() - t0)
        return _CmdResult(
            cmd=str(cmd),
            cwd=str(cwd),
            ok=bool(ok),
            returncode=int(rc),
            duration_sec=float(dt),
            timed_out=bool(timed_out),
            stdout_tail=_tail(out),
            stderr_tail=_tail(err),
        )

    async def run(self, *, manifest: Manifest, manifest_path: Path | None) -> dict[str, Any]:
        start = time.monotonic()
        deadline = float(start + float(self.max_wall_time_sec))
        now = datetime.now(timezone.utc)

        payload: dict[str, Any] = {
            "ok": True,
            "process": "idle",
            "created_at": now.isoformat(),
            "max_wall_time_sec": int(self.max_wall_time_sec),
            "steps": [],
            "budget_exhausted": False,
        }

        # --- Readiness: code graph sync ---
        if self.run_code_graph_sync:
            if self._time_left(deadline=deadline) <= 0:
                payload["budget_exhausted"] = True
            else:
                try:
                    sync = CodeGraphSync(
                        agents=self.agents,
                        agent_key=self.code_graph_sync_agent,
                        index_namespace=self.index_namespace,
                    )
                    res = await sync.run(manifest=manifest, manifest_path=manifest_path)
                    payload["steps"].append(
                        {
                            "name": "code_graph_sync",
                            "ok": bool(res.get("ok", False)),
                            "result": res,
                        }
                    )
                except Exception as e:
                    payload["ok"] = False
                    payload["steps"].append(
                        {
                            "name": "code_graph_sync",
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

        # --- Continuous evaluation: short, deterministic commands ---
        if self.run_eval and self.eval_cmds and not bool(payload.get("budget_exhausted", False)):
            cwd = Path(self.eval_cwd)
            # Treat eval_cwd as relative to the current process working directory.
            if not cwd.is_absolute():
                cwd = (Path.cwd() / cwd).resolve()
            eval_results: list[dict[str, Any]] = []
            for cmd in self.eval_cmds:
                left = self._time_left(deadline=deadline)
                if left <= 0:
                    payload["budget_exhausted"] = True
                    break
                # Respect both per-command and overall budgets.
                timeout = int(min(float(self.eval_timeout_sec), float(left)))
                r = self._run_cmd(cmd=str(cmd), cwd=cwd, timeout_sec=timeout)
                eval_results.append(
                    {
                        "cmd": r.cmd,
                        "cwd": r.cwd,
                        "ok": bool(r.ok),
                        "returncode": int(r.returncode),
                        "duration_sec": float(r.duration_sec),
                        "timed_out": bool(r.timed_out),
                        "stdout_tail": r.stdout_tail,
                        "stderr_tail": r.stderr_tail,
                    }
                )
                if not r.ok:
                    # Stop early on failed eval: this is meant to be a guardrail.
                    break
            payload["steps"].append(
                {
                    "name": "eval",
                    "ok": bool(all(bool(x.get("ok", False)) for x in eval_results)),
                    "results": eval_results,
                }
            )
            if eval_results and not bool(eval_results[-1].get("ok", False)):
                payload["ok"] = False

        # --- Research loop: short iteration ---
        if self.run_research_loop and not bool(payload.get("budget_exhausted", False)):
            if self._time_left(deadline=deadline) <= 0:
                payload["budget_exhausted"] = True
            else:
                try:
                    loop = ResearchLoopProcess(
                        agents=self.agents,
                        leader_key=self.leader_key,
                        writer_key=self.writer_key,
                        reviewer_key=self.reviewer_key,
                        max_iterations=int(self.research_max_iterations),
                        auto_run_experiments=bool(self.research_auto_run_experiments),
                        output_dir=str(self.output_dir),
                    )
                    res2 = await loop.run(manifest=manifest, manifest_path=manifest_path)
                    payload["steps"].append(
                        {
                            "name": "research_loop",
                            "ok": bool(res2.get("ok", False)),
                            "result": res2,
                        }
                    )
                    if not bool(res2.get("ok", False)):
                        payload["ok"] = False
                except Exception as e:
                    payload["ok"] = False
                    payload["steps"].append(
                        {
                            "name": "research_loop",
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

        elapsed = float(time.monotonic() - start)
        payload["elapsed_sec"] = float(elapsed)
        payload["budget_exhausted"] = bool(payload.get("budget_exhausted", False) or elapsed >= float(self.max_wall_time_sec))
        try:
            logger.success(
                f"idle: completed in {elapsed:.2f}s (budget={self.max_wall_time_sec}s, exhausted={payload['budget_exhausted']})"
            )
        except Exception:
            pass
        return payload

