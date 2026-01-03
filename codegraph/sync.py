from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from falkordb import FalkorDB, Graph

from caramba.console import logger
from caramba.codegraph.parser import Edge, Node


def _connect_falkordb(
    *,
    uri: str | None = None,
    host: str | None = None,
    port: int | None = None,
    password: str | None = None,
) -> FalkorDB:
    uri = uri or os.getenv("FALKORDB_URI") or os.getenv("FALKOR_URI")
    if uri:
        return FalkorDB(url=uri, password=password or os.getenv("FALKORDB_PASSWORD") or None)
    return FalkorDB(
        host=host or os.getenv("FALKORDB_HOST") or "localhost",
        port=int(port or os.getenv("FALKORDB_PORT") or 6379),
        password=password or os.getenv("FALKORDB_PASSWORD") or None,
    )


def _graph_name(default: str = "caramba_code") -> str:
    return os.getenv("CARAMBA_CODEGRAPH_NAME") or default


def _delete_file_scope(g: Graph, *, file: str) -> None:
    # Best-effort: remove all nodes anchored to this file.
    # (This also removes relationships for those nodes.)
    g.query("MATCH (n) WHERE n.file = $file DETACH DELETE n", {"file": file})


def _upsert_nodes(g: Graph, label: str, nodes: list[Node]) -> None:
    if not nodes:
        return
    payload = [{"id": n.id, **n.props} for n in nodes]
    q = (
        f"UNWIND $rows AS row "
        f"MERGE (n:{label} {{id: row.id}}) "
        f"SET n += row"
    )
    g.query(q, {"rows": payload})


def _upsert_edges(g: Graph, rel: str, edges: list[Edge]) -> None:
    if not edges:
        return
    payload = [{"src": e.src, "dst": e.dst, "props": e.props} for e in edges]
    q = (
        "UNWIND $rows AS row "
        "MATCH (a {id: row.src}) "
        "MATCH (b {id: row.dst}) "
        f"MERGE (a)-[r:{rel}]->(b) "
        "SET r += row.props"
    )
    g.query(q, {"rows": payload})


def sync_files_to_falkordb(
    *,
    repo_root: str | Path,  # noqa: ARG001
    nodes: list[Node],
    edges: list[Edge],
    files: list[str] | None,
    graph: str | None = None,
    uri: str | None = None,
    host: str | None = None,
    port: int | None = None,
    password: str | None = None,
    reset: bool = False,
    best_effort: bool = True,
) -> dict[str, Any]:
    """Sync parsed nodes/edges into FalkorDB.

    Strategy:
    - If `reset`: wipe the whole graph.
    - Else: for each file in `files`, delete nodes scoped to that file then upsert.

    Note: `repo_root` is reserved for future use (e.g., incremental sync by repo scope).
    """
    try:
        client = _connect_falkordb(uri=uri, host=host, port=port, password=password)
        gname = graph or _graph_name()
        g = Graph(client, gname)

        if reset:
            g.query("MATCH (n) DETACH DELETE n")

        # If a file list is provided, scope deletes to those files. Otherwise assume full rebuild.
        if files:
            for f in sorted(set(files)):
                _delete_file_scope(g, file=f)

        by_label: dict[str, list[Node]] = defaultdict(list)
        for n in nodes:
            by_label[n.label].append(n)

        # Upsert nodes by label
        for label, items in by_label.items():
            _upsert_nodes(g, label, items)

        by_rel: dict[str, list[Edge]] = defaultdict(list)
        for e in edges:
            by_rel[e.rel].append(e)

        for rel, items in by_rel.items():
            _upsert_edges(g, rel, items)

        return {
            "ok": True,
            "graph": gname,
            "nodes": len(nodes),
            "edges": len(edges),
            "files": len(files or []),
            "reset": bool(reset),
        }
    except Exception as e:
        if not best_effort:
            raise
        logger.warning(f"codegraph: failed to sync to FalkorDB (best-effort): {e}")
        return {"ok": False, "error": str(e), "best_effort": True}

