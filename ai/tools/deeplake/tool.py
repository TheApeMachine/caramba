"""Deep Lake + ColBERT (late interaction).

Implements the Deep Lake "ColBERT: late interaction" approach:
- store 2D passage embeddings (per-doc token vectors)
- query with `maxsim(embedding, query_matrix)` in TQL

Reference:
- https://docs.deeplake.ai/latest/guide/rag/#6-colbert-efficient-and-effective-passage-search-via-contextualized-late-interaction-over-bert
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence
import deeplake
from deeplake import types
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from console import logger
_logger = logger


def _rowview_to_str_dict(row: object) -> dict[str, Any]:
    """Convert a DeepLake RowView-like object into a str-keyed dict.

    DeepLake rows are mapping-like but often typed as bytes-keyed (e.g. RowView[bytes, ...]),
    which makes static type checkers reject `row.get("column")` and `dict(row)`.
    """
    if isinstance(row, dict):
        out_dict: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(k, bytes):
                out_dict[k.decode("utf-8", errors="replace")] = v
            else:
                out_dict[str(k)] = v
        return out_dict

    # RowView: supports keys() and __getitem__ access.
    if hasattr(row, "keys") and hasattr(row, "__getitem__"):
        try:
            keys = list(row.keys())  # type: ignore[attr-defined]
        except Exception:
            return {}

        out_row: dict[str, Any] = {}
        for k in keys:
            sk = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)
            try:
                out_row[sk] = row[k]  # type: ignore[index]
            except Exception:
                # Non-critical; skip columns we can't read.
                continue
        return out_row

    return {}


# Initialize FastMCP server (Docker-friendly HTTP transport)
mcp = FastMCP("DeepLake Tools", json_response=True)
mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)


class DeepLakeTool():
    """DeepLake tool"""
    def __init__(
        self,
        *,
        dataset_uri: str | None = None,
        dataset_dir: str | Path | None = None,
        model_name: str = "colbert-ir/colbertv2.0",
        colbert_root: str | Path = "experiments",
        create_if_missing: bool = True,
        embedding_column: str = "embedding",
        text_column: str = "text",
        id_column: str = "id",
        metadata_column: str = "metadata_json",
    ):
        if dataset_uri is None:
            if dataset_dir is None:
                dataset_dir = Path(".caramba") / "deeplake_colbert"

            dataset_uri = f"file://{Path(dataset_dir).expanduser().resolve()}"

        self.dataset_uri = dataset_uri
        self.embedding_column = embedding_column
        self.text_column = text_column
        self.id_column = id_column
        self.metadata_column = metadata_column

        self.checkpoint = Checkpoint(model_name, colbert_config=ColBERTConfig(root=str(colbert_root)))
        self.ds = self.open_or_create(create_if_missing=create_if_missing)

    def get_command(self) -> str:
        return sys.executable

    def get_args(self) -> list[str]:
        """Get the arguments for the tool."""
        return ["-m", "ai.tools.deeplake"]

    def open_or_create(self, *, create_if_missing: bool) -> deeplake.core.dataset.Dataset:
        """Open or create the DeepLake dataset."""
        try:
            ds = deeplake.core.dataset.Dataset(self.dataset_uri)
        except Exception as e:
            # Check if this is a missing dataset error
            error_msg = str(e).lower()
            is_missing = (
                "not found" in error_msg
                or "does not exist" in error_msg
                or isinstance(e, FileNotFoundError)
            )
            if not create_if_missing or not is_missing:
                raise
            ds = deeplake.create(self.dataset_uri)

        existing = {c.name for c in getattr(ds, "columns", [])} if hasattr(ds, "columns") else set()
        if self.id_column not in existing:
            ds.add_column(name=self.id_column, dtype=types.Text())
        if self.text_column not in existing:
            ds.add_column(name=self.text_column, dtype=types.Text())
        if self.metadata_column not in existing:
            ds.add_column(name=self.metadata_column, dtype=types.Text())
        if self.embedding_column not in existing:
            ds.add_column(name=self.embedding_column, dtype=types.Array("float32", dimensions=2))
        ds.commit()
        return ds

    def embed_documents(self, texts: list[str]) -> list[list[list[float]]]:
        """Embed documents using the ColBERT model."""
        if not hasattr(self.checkpoint, "docFromText"):
            raise RuntimeError("Unsupported colbert-ai API: expected Checkpoint.docFromText().")
        mats = self.checkpoint.docFromText(texts)
        return [m.tolist() for m in mats]  # type: ignore[attr-defined]

    def embed_query(self, query: str) -> list[list[float]]:
        """Embed a query using the ColBERT model."""
        if not hasattr(self.checkpoint, "queryFromText"):
            raise RuntimeError("Unsupported colbert-ai API: expected Checkpoint.queryFromText().")
        mat = self.checkpoint.queryFromText([query])[0]
        return mat.tolist()

    def search(self, query: str) -> list[dict]:
        """Search the DeepLake database."""
        q_mat = self.embed_query(query)
        q_str = self.matrix_to_tql_array(q_mat)

        tql = f"""
            SELECT *,
                   maxsim({self.embedding_column}, {q_str}) as score
            ORDER BY maxsim({self.embedding_column}, {q_str}) DESC
            LIMIT 10
        """
        view = self.ds.query(tql)

        out: list[dict] = []
        for row in view:
            # Use getattr with default to avoid expensive exception handling in loops.
            # DeepLake rows support dict-like access, so we use a helper approach.
            row_dict = _rowview_to_str_dict(row)

            raw = row_dict.get(self.metadata_column, "{}")
            if isinstance(raw, str):
                try:
                    meta = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    meta = {}
            else:
                meta = {}

            row_id = str(row_dict.get(self.id_column, ""))
            row_text = str(row_dict.get(self.text_column, ""))

            score_val = row_dict.get("score", 0.0)
            try:
                row_score = float(score_val) if score_val is not None else 0.0
            except (TypeError, ValueError):
                row_score = 0.0

            out.append(
                {
                    "id": row_id,
                    "text": row_text,
                    "score": row_score,
                    "metadata": meta,
                }
            )
        return out

    def store(self, data: dict) -> str:
        """Store data in the DeepLake database."""
        items = data.get("items")

        if items is None:
            items = [data]

        if not isinstance(items, list):
            raise TypeError("DeepLakeTool.store expects a dict with optional 'items': list[dict].")

        ids: list[str] = []
        texts: list[str] = []
        metas: list[str] = []

        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                _logger.warning(f"Skipping non-dict item at index {idx}: {repr(it)}")
                continue

            ids.append(str(it.get("id", "")))
            texts.append(str(it.get("text", "")))
            metas.append(json.dumps(it.get("metadata", {}), ensure_ascii=False))

        matrices = self.embed_documents(texts)

        self.ds.append(
            {
                self.id_column: ids,
                self.text_column: texts,
                self.metadata_column: metas,
                self.embedding_column: matrices,
            }
        )
        self.ds.commit()

        return "Data stored successfully."

    def matrix_to_tql_array(self, matrix: Sequence[Sequence[float]]) -> str:
        rows: list[str] = []
        for row in matrix:
            rows.append("ARRAY[" + ",".join(str(float(x)) for x in row) + "]")
        return "ARRAY[" + ",".join(rows) + "]"


_DEEPLAKE_TOOL: DeepLakeTool | None = None


def _cleanup_deeplake_tool() -> None:
    """Clean up the global DeepLake tool singleton.

    Call this during application shutdown to properly close resources.
    """
    global _DEEPLAKE_TOOL
    if _DEEPLAKE_TOOL is not None:
        try:
            # DeepLake datasets should be committed before closing.
            if hasattr(_DEEPLAKE_TOOL, 'ds') and _DEEPLAKE_TOOL.ds is not None:
                try:
                    _DEEPLAKE_TOOL.ds.commit()
                except Exception as e:
                    _logger.error(f"Failed to commit DeepLake dataset: {e}")
        except Exception as e:
            _logger.error(f"Failed to cleanup DeepLake tool: {e}")
        _DEEPLAKE_TOOL = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _get_tool() -> DeepLakeTool:
    """Lazy singleton DeepLakeTool for MCP requests.

    Environment variables:
      - DEEPLAKE_DATASET_URI: explicit dataset URI (e.g. "file:///app/.caramba/deeplake_colbert")
      - DEEPLAKE_DATASET_DIR: local directory for file:// datasets (defaults to ".caramba/deeplake_colbert")
      - DEEPLAKE_MODEL_NAME: ColBERT model id (default: "colbert-ir/colbertv2.0")
      - DEEPLAKE_COLBERT_ROOT: ColBERT experiments root (default: "experiments")
      - DEEPLAKE_CREATE_IF_MISSING: create dataset if missing (default: true)
    """
    global _DEEPLAKE_TOOL
    if _DEEPLAKE_TOOL is not None:
        return _DEEPLAKE_TOOL

    dataset_uri = os.getenv("DEEPLAKE_DATASET_URI") or None
    dataset_dir = os.getenv("DEEPLAKE_DATASET_DIR") or None
    model_name = os.getenv("DEEPLAKE_MODEL_NAME", "colbert-ir/colbertv2.0")
    colbert_root = os.getenv("DEEPLAKE_COLBERT_ROOT", "experiments")
    create_if_missing = _env_bool("DEEPLAKE_CREATE_IF_MISSING", True)

    _DEEPLAKE_TOOL = DeepLakeTool(
        dataset_uri=dataset_uri,
        dataset_dir=dataset_dir,
        model_name=model_name,
        colbert_root=colbert_root,
        create_if_missing=create_if_missing,
    )
    return _DEEPLAKE_TOOL


@mcp.tool()
def deeplake_store(data: dict) -> str:
    """Store one or many items in DeepLake.

    Expected shape:
      - {"id": "...", "text": "...", "metadata": {...}}
    Or:
      - {"items": [ {id,text,metadata}, ... ]}
    """
    return _get_tool().store(data)


@mcp.tool()
def deeplake_search(query: str) -> list[dict]:
    """Semantic search over stored items (top-10)."""
    return _get_tool().search(query)


# Backward compatibility wrapper for stdio transport / tool spawning
class DeepLakeMCP:
    """DeepLake MCP server (backward compatibility wrapper)."""

    def __init__(self, **kwargs: object):
        # Present for compatibility with older codepaths; runtime config is via env vars.
        pass

    def get_command(self) -> str:
        return sys.executable

    def get_args(self) -> list[str]:
        return ["-m", "ai.tools.deeplake"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    app = mcp.streamable_http_app()

    def root(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    app.add_route("/", root, methods=["GET"])
    app.add_route("/health", health, methods=["GET"])

    uvicorn.run(app, host=args.host, port=args.port)
