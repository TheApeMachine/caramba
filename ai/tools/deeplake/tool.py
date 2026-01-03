"""Deep Lake + ColBERT (late interaction).

Implements the Deep Lake "ColBERT: late interaction" approach:
- store 2D passage embeddings (per-doc token vectors)
- query with `maxsim(embedding, query_matrix)` in TQL

Reference:
- https://docs.deeplake.ai/latest/guide/rag/#6-colbert-efficient-and-effective-passage-search-via-contextualized-late-interaction-over-bert
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence
import deeplake
from deeplake import types
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


# Initialize FastMCP server (Docker-friendly HTTP transport)
mcp = FastMCP("DeepLake Tools", json_response=True)


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

    def open_or_create(self, *, create_if_missing: bool) -> deeplake.Dataset:
        """Open or create the DeepLake dataset."""
        try:
            ds = deeplake.open(self.dataset_uri)
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
            try:
                raw = row[self.metadata_column]  # type: ignore[index]
            except (KeyError, IndexError):
                raw = "{}"
            try:
                meta = json.loads(raw) if isinstance(raw, str) else {}
            except Exception:
                meta = {}
            try:
                row_id = str(row[self.id_column])  # type: ignore[index]
            except (KeyError, IndexError):
                row_id = ""
            try:
                row_text = str(row[self.text_column])  # type: ignore[index]
            except (KeyError, IndexError):
                row_text = ""
            try:
                row_score = float(row["score"])  # type: ignore[index]
            except (KeyError, IndexError, ValueError):
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
                logger.warning(f"Skipping non-dict item at index {idx}: {repr(it)}")
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
    # Host/port are controlled via MCP_SERVER_HOST / MCP_SERVER_PORT env vars.
    mcp.run(transport="streamable-http")
