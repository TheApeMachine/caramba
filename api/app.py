"""FastAPI control-plane for Caramba (dev / local use).

This replaces the earlier stdlib HTTPServer implementation with a standard API stack:
- FastAPI request/response models
- uvicorn server
- robust async streaming (SSE)

The goal is to bridge the frontend to real caramba runs:
- spawn manifest targets (`python -m caramba run ...`)
- stream training metrics from `train.jsonl`
- stream process logs
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config.manifest import Manifest
from config.model import ModelConfig
from experiment.runner import _resolve_target as _resolve_target_name
from console import logger

_REPO_ROOT = Path(__file__).resolve().parent


class StartRunRequest(BaseModel):
    manifest_path: str = Field(..., min_length=1)
    target: str | None = None


class RunPublic(BaseModel):
    id: str
    manifest_path: str
    target: str
    cmd: list[str]
    cwd: str
    pid: int
    started_at_s: float
    run_dir: str
    jsonl_path: str
    log_path: str
    returncode: int | None = None
    ended_at_s: float | None = None


@dataclass(slots=True)
class RunRecord:
    id: str
    manifest_path: str
    target: str
    cmd: list[str]
    cwd: str
    pid: int
    started_at_s: float
    run_dir: str
    jsonl_path: str
    log_path: str
    returncode: int | None = None
    ended_at_s: float | None = None
    proc: asyncio.subprocess.Process | None = None
    log_fh: Any | None = None

    def public(self) -> RunPublic:
        return RunPublic(
            id=self.id,
            manifest_path=self.manifest_path,
            target=self.target,
            cmd=list(self.cmd),
            cwd=self.cwd,
            pid=int(self.pid),
            started_at_s=float(self.started_at_s),
            run_dir=self.run_dir,
            jsonl_path=self.jsonl_path,
            log_path=self.log_path,
            returncode=self.returncode,
            ended_at_s=self.ended_at_s,
        )


_RUNS: dict[str, RunRecord] = {}
_RUNS_LOCK = asyncio.Lock()
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _resolve_manifest_path(s: str) -> Path:
    p = Path(s)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


async def get_manifest(path: str) -> Manifest:
    """FastAPI dependency to load and validate a manifest from a path."""
    mp = _resolve_manifest_path(path)
    if not mp.exists():
        raise HTTPException(status_code=400, detail={"error": "manifest_not_found", "path": str(mp)})
    try:
        return Manifest.from_path(mp)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "manifest_invalid", "detail": str(e)}) from e


def _resolve_model_config(manifest: Manifest, target: str) -> tuple[str, ModelConfig]:
    tgt = next((t for t in manifest.targets if getattr(t, "name", None) == target), None)
    if tgt is None:
        raise HTTPException(status_code=404, detail={"error": "target_not_found", "target": target})

    system = getattr(tgt, "system", None)
    sys_ref = getattr(system, "ref", "")
    sys_cfg = getattr(system, "config", {}) or {}
    if sys_ref not in ("system.language_model", "system.generic"):
        raise HTTPException(status_code=400, detail={"error": "unsupported_system", "system_ref": str(sys_ref)})
    model_payload = sys_cfg.get("model", None) if isinstance(sys_cfg, dict) else None
    if not isinstance(model_payload, dict):
        raise HTTPException(status_code=400, detail={"error": "missing_model_payload"})

    cfg = ModelConfig.model_validate(model_payload)
    return str(sys_ref), cfg


def _target_info(manifest: Manifest, target: object) -> dict[str, object]:
    """Metadata for a manifest target.

    This is used by the UI to avoid calling model-only endpoints for process-only targets.
    """

    name = getattr(target, "name", None)
    type_ = getattr(target, "type", None)
    system = getattr(target, "system", None)
    sys_ref = getattr(system, "ref", None)
    sys_cfg = getattr(system, "config", None)

    model_capable = False
    if sys_ref in ("system.language_model", "system.generic") and isinstance(sys_cfg, dict):
        model_payload = sys_cfg.get("model", None)
        model_capable = isinstance(model_payload, dict)

    if type_ is None:
        type_s = None
    else:
        type_s = getattr(type_, "value", None)
        if not isinstance(type_s, str):
            type_s = str(type_)

    return {
        "name": str(name) if name is not None else None,
        "type": type_s,
        "system_ref": str(sys_ref) if sys_ref is not None else None,
        "model_capable": bool(model_capable),
    }


def _walk_model_topology(node: object) -> list[object]:
    """Recursively walk a model topology configuration to extract all layer nodes."""
    # Important: use the same module path as ModelConfig's parsing types.
    # `config.*` and `caramba.config.*` can be distinct modules at runtime, which
    # breaks isinstance checks and causes the UI to think there is only 1 layer.
    from config.topology import GraphTopologyConfig

    if isinstance(node, GraphTopologyConfig):
        return []
    layers = getattr(node, "layers", None)
    if isinstance(layers, list):
        repeat = int(getattr(node, "repeat", 1) or 1)
        repeat = max(1, repeat)
        out: list[object] = []
        for _ in range(repeat):
            for child in layers:
                out.extend(_walk_model_topology(child))
        return out
    return [node]


async def _watch_proc(run_id: str) -> None:
    async with _RUNS_LOCK:
        rec = _RUNS.get(run_id)
    if rec is None or rec.proc is None:
        return
    try:
        rc = await rec.proc.wait()
    except Exception as e:
        logger.error(f"Failed to wait for process: {e}")
        rc = -1
    async with _RUNS_LOCK:
        r = _RUNS.get(run_id)
        if r is None:
            return
        r.returncode = int(rc)
        r.ended_at_s = float(time.time())
        fh = r.log_fh
        r.log_fh = None

    if fh is not None:
        try:
            await asyncio.get_running_loop().run_in_executor(None, fh.close)
        except Exception as e:
            logger.error(f"Failed to close log file: {e}")


async def _spawn_run(*, manifest_path: Path, target_arg: str | None) -> RunRecord:
    manifest = Manifest.from_path(manifest_path)
    target_name = _resolve_target_name(manifest, target_arg)

    run_id = uuid.uuid4().hex
    run_dir = (_REPO_ROOT / "runs" / str(target_name)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_dir / "train.jsonl"
    log_path = run_dir / f"server_{run_id}.log"
    loop = asyncio.get_running_loop()
    log_fh = await loop.run_in_executor(
        None,
        lambda: open(str(log_path), "a", encoding="utf-8", buffering=1),
    )

    cmd = [
        sys.executable,
        "-m",
        "caramba",
        "run",
        str(manifest_path),
        "--target",
        str(target_name),
    ]

    # Create subprocess with stdout/stderr redirected to the log file.
    # asyncio only supports file descriptors for stdout/stderr redirection.
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_REPO_ROOT),
            stdout=log_fh.fileno(),
            stderr=log_fh.fileno(),
            env=os.environ.copy(),
            # Ensure the run has its own process group so "Stop" can reliably
            # terminate the whole job (and any children).
            start_new_session=True,
        )
    except Exception:
        await loop.run_in_executor(None, log_fh.close)
        raise

    rec = RunRecord(
        id=str(run_id),
        manifest_path=str(manifest_path),
        target=str(target_name),
        cmd=[str(x) for x in cmd],
        cwd=str(_REPO_ROOT),
        pid=int(proc.pid),
        started_at_s=float(time.time()),
        run_dir=str(run_dir),
        jsonl_path=str(jsonl_path),
        log_path=str(log_path),
        proc=proc,
        log_fh=log_fh,
    )

    async with _RUNS_LOCK:
        _RUNS[rec.id] = rec
    task = asyncio.create_task(_watch_proc(rec.id))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return rec


async def _stop_run(rec: RunRecord, *, timeout_s: float = 5.0) -> bool:
    proc = rec.proc
    pid = int(rec.pid)
    if proc is None:
        # Terminate by PID when we don't own the process handle.
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except Exception as e:
            logger.warning(f"StopRun: Failed to terminate process {pid}: {e}")
            return False
        return True

    if proc.returncode is not None:
        return True
    # Prefer killing the whole process group when possible (macOS/Linux).
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        # Fall back to terminating just the parent process.
        try:
            proc.terminate()
        except ProcessLookupError:
            logger.error("Process not found")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate process: {e}")
    try:
        await asyncio.wait_for(proc.wait(), timeout=float(timeout_s))
        return True
    except asyncio.TimeoutError:
        pass
    try:
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        return True
    except Exception as e:
        logger.error(f"Failed to kill process: {e}")
        return False


def _sse_json(obj: dict[str, Any]) -> bytes:
    # Standard SSE "data:" framing (single line JSON).
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode("utf-8")


async def _tail_file_sse(
    *,
    path: Path,
    event_type: Literal["log", "raw"],
    from_end: bool,
    run_id: str | None,
    poll_s: float,
) -> AsyncIterator[bytes]:
    # Initial comment opens the stream.
    yield b": ok\n\n"
    # Wait for file.
    while not path.exists():
        yield _sse_json(
            {
                "type": "server",
                "ts": time.time(),
                "run_id": run_id,
                "phase": None,
                "step": None,
                "data": {"status": "waiting_for_file", "path": str(path)},
            }
        )
        await asyncio.sleep(max(0.1, poll_s))

    # Tail.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if from_end:
            f.seek(0, os.SEEK_END)

        last_keepalive = time.time()
        while True:
            line = f.readline()
            if line:
                s = line.rstrip("\n").rstrip("\r")
                if s:
                    if event_type == "log":
                        yield _sse_json(
                            {
                                "type": "log",
                                "ts": time.time(),
                                "run_id": run_id,
                                "phase": None,
                                "step": None,
                                "data": {"line": s},
                            }
                        )
                    else:
                        # "raw" means the line is already jsonl
                        yield f"data: {s}\n\n".encode("utf-8")
                continue

            now = time.time()
            if now - last_keepalive >= 5.0:
                yield b": keep-alive\n\n"
                last_keepalive = now
            await asyncio.sleep(poll_s)


def create_app() -> FastAPI:
    app = FastAPI(title="caramba control-plane", version="0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/healthz")
    async def healthz() -> dict[str, object]:
        return {"ok": True, "ts": time.time()}

    @app.post("/api/runs")
    async def start_run(req: StartRunRequest) -> JSONResponse:
        mp = _resolve_manifest_path(req.manifest_path)
        if not mp.exists():
            raise HTTPException(status_code=400, detail={"error": "manifest_not_found", "path": str(mp)})
        try:
            rec = await _spawn_run(manifest_path=mp, target_arg=req.target)
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": "spawn_failed", "detail": str(e)}) from e
        return JSONResponse({"run": rec.public().model_dump()})

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        async with _RUNS_LOCK:
            rec = _RUNS.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail={"error": "unknown_run", "run_id": run_id})
        return JSONResponse({"run": rec.public().model_dump()})

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(run_id: str) -> JSONResponse:
        async with _RUNS_LOCK:
            rec = _RUNS.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail={"error": "unknown_run", "run_id": run_id})
        ok = await _stop_run(rec)
        return JSONResponse({"ok": bool(ok), "run_id": run_id})

    @app.get("/api/runs/{run_id}/logs")
    async def stream_logs(run_id: str, from_: Literal["start", "end"] = "start", poll_ms: int = 200) -> StreamingResponse:
        async with _RUNS_LOCK:
            rec = _RUNS.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail={"error": "unknown_run", "run_id": run_id})
        poll_s = max(0.05, min(2.0, float(poll_ms) / 1000.0))
        gen = _tail_file_sse(
            path=Path(rec.log_path),
            event_type="log",
            from_end=(from_ == "end"),
            run_id=rec.id,
            poll_s=poll_s,
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str, from_: Literal["start", "end"] = "end", poll_ms: int = 200) -> StreamingResponse:
        async with _RUNS_LOCK:
            rec = _RUNS.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail={"error": "unknown_run", "run_id": run_id})
        poll_s = max(0.05, min(2.0, float(poll_ms) / 1000.0))
        gen = _tail_file_sse(
            path=Path(rec.jsonl_path),
            event_type="raw",
            from_end=(from_ == "end"),
            run_id=None,
            poll_s=poll_s,
        )
        return StreamingResponse(gen, media_type="text/event-stream")

    @app.get("/api/manifests/targets")
    async def manifest_targets(path: str, manifest: Manifest = Depends(get_manifest)) -> dict[str, object]:
        mp = _resolve_manifest_path(path)
        entries = [_target_info(manifest, t) for t in manifest.targets]
        targets_all = [e["name"] for e in entries if isinstance(e.get("name"), str)]
        model_targets = [
            e["name"]
            for e in entries
            if isinstance(e.get("name"), str) and bool(e.get("model_capable"))
        ]
        process_targets = [
            e["name"]
            for e in entries
            if isinstance(e.get("name"), str) and e.get("type") == "process"
        ]
        return {
            "manifest_path": str(mp),
            "targets": targets_all,
            "model_targets": model_targets,
            "process_targets": process_targets,
            "entries": entries,
        }

    @app.get("/api/manifests/model_summary")
    async def model_summary(path: str, target: str, manifest: Manifest = Depends(get_manifest)) -> dict[str, object]:
        mp = _resolve_manifest_path(path)
        sys_ref, cfg = _resolve_model_config(manifest, target)

        from config.layer import AttentionLayerConfig

        nodes = _walk_model_topology(cfg.topology)
        attn_layers = [n for n in nodes if isinstance(n, AttentionLayerConfig)]
        n_layers = len(attn_layers)
        d_model = int(attn_layers[0].d_model) if attn_layers else None
        n_heads = int(attn_layers[0].n_heads) if attn_layers else None
        mode = str(getattr(attn_layers[0], "mode", "")).lower() if attn_layers else None

        vocab_size = None
        try:
            emb = getattr(cfg, "embedder", None)
            vocab_size = int(getattr(emb, "vocab_size", 0)) or None
        except Exception:
            vocab_size = None

        return {
            "manifest_path": str(mp),
            "target": str(target),
            "system_ref": str(sys_ref),
            "model": {
                "type": str(cfg.type.value),
                "tied_embeddings": bool(getattr(cfg, "tied_embeddings", True)),
                "vocab_size": vocab_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": int(n_layers),
                "attention_mode": mode,
            },
        }

    @app.get("/api/manifests/attention_layers")
    async def attention_layers(path: str, target: str, manifest: Manifest = Depends(get_manifest)) -> dict[str, object]:
        mp = _resolve_manifest_path(path)
        sys_ref, cfg = _resolve_model_config(manifest, target)

        from config.layer import AttentionLayerConfig

        nodes = _walk_model_topology(cfg.topology)
        attn_cfgs = [n for n in nodes if isinstance(n, AttentionLayerConfig)]
        out_layers: list[dict[str, object]] = []
        for i, a in enumerate(attn_cfgs):
            out_layers.append(
                {
                    "index": int(i),
                    "mode": str(getattr(a, "mode", "")).lower(),
                    "d_model": int(getattr(a, "d_model", 0)),
                    "n_heads": int(getattr(a, "n_heads", 0)),
                    "n_kv_heads": int(getattr(a, "kv_heads", 0)),
                    "attn_dim": int(getattr(a, "attn_dim", 0)) if getattr(a, "attn_dim", None) is not None else None,
                    "sem_dim": int(getattr(a, "sem_dim", 0)) if getattr(a, "sem_dim", None) is not None else None,
                    "geo_dim": int(getattr(a, "geo_dim", 0)) if getattr(a, "geo_dim", None) is not None else None,
                    "rope_enabled": bool(getattr(a, "rope_enabled", False)),
                    "rope_base": float(getattr(a, "rope_base", 0.0)),
                    "rope_semantic": bool(getattr(a, "rope_semantic", False)),
                    "tie_qk": bool(getattr(a, "tie_qk", False)),
                    "null_attn": bool(getattr(a, "null_attn", False)),
                    "decoupled_gate": bool(getattr(a, "decoupled_gate", False)),
                }
            )

        return {
            "manifest_path": str(mp),
            "target": str(target),
            "system_ref": str(sys_ref),
            "count": len(out_layers),
            "layers": out_layers,
        }

    return app


app = create_app()
