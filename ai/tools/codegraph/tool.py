"""Code graph tool (FalkorDB-backed).

Run this as a standalone MCP SSE server:
    python -m ai.tools.codegraph.tool

Then connect to it via MCP SSE at:
    http://localhost:<port>/sse
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

import uvicorn
from falkordb import FalkorDB, Graph
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


mcp = FastMCP("CodeGraph Tool", json_response=True)


_WRITE_KEYWORDS = re.compile(
    r"(?is)\b("
    r"create|merge|delete|detach\s+delete|set|remove|foreach|load\s+csv|commit|rollback|drop|alter"
    r")\b"
)
_CALL_KEYWORD = re.compile(r"(?is)\bcall\b")


def _graph_name() -> str:
    return os.getenv("CARAMBA_CODEGRAPH_NAME") or "caramba_code"


def _connect() -> Graph:
    uri = os.getenv("FALKORDB_URI") or os.getenv("FALKOR_URI") or "redis://localhost:6379"
    pwd = os.getenv("FALKORDB_PASSWORD") or None
    client = FalkorDB(url=uri, password=pwd)
    return Graph(client, _graph_name())


def _is_safe_readonly_cypher(q: str) -> bool:
    s = (q or "").strip()
    if not s:
        return False
    if _WRITE_KEYWORDS.search(s):
        return False
    # Disallow procedures by default (many can mutate state or do arbitrary work).
    if _CALL_KEYWORD.search(s):
        return False
    return True


def _limit_rows(rows: list[list[object]], limit: int) -> list[list[object]]:
    limit = max(1, min(int(limit or 50), 500))
    return rows[:limit]


def _coerce_result(res: Any, *, limit: int) -> dict[str, Any]:
    header = getattr(res, "header", None)
    result_set = getattr(res, "result_set", None)
    cols: list[str] = []
    if isinstance(header, list):
        # header items are typically column names (strings)
        cols = [str(x) for x in header]
    rows: list[list[object]] = []
    if isinstance(result_set, list):
        for r in result_set:
            if isinstance(r, (list, tuple)):
                rows.append(list(r))
            else:
                rows.append([r])
    rows = _limit_rows(rows, limit)
    return {"columns": cols, "rows": rows, "row_count": len(rows)}


@mcp.tool()
def ping() -> dict[str, Any]:
    """Check connectivity to the code graph."""
    g = _connect()
    # minimal query
    res = g.query("MATCH (n) RETURN count(n) AS n LIMIT 1")
    out = _coerce_result(res, limit=1)
    return {"ok": True, "graph": _graph_name(), "stats": out}


@mcp.tool()
def query(cypher: str, params: dict[str, object] | None = None, limit: int = 50) -> dict[str, Any]:
    """Run a READ-ONLY Cypher query against the code graph.

    Safety:
    - rejects write keywords (CREATE/MERGE/DELETE/SET/...)
    - rejects CALL (procedures)
    """
    if not _is_safe_readonly_cypher(cypher):
        return {
            "ok": False,
            "error": "unsafe_query",
            "hint": "Only read-only MATCH/RETURN queries are allowed (no CALL/CREATE/MERGE/DELETE/SET/etc).",
        }
    g = _connect()
    res = g.query(cypher, params=params or {})
    return {"ok": True, "graph": _graph_name(), "result": _coerce_result(res, limit=limit)}


@mcp.tool()
def neighbors(node_id: str, rel: str | None = None, direction: str = "out", limit: int = 50) -> dict[str, Any]:
    """Return neighbors of a node by `id` (optionally filter by relationship type)."""
    direction = (direction or "out").strip().lower()
    if direction not in {"out", "in", "both"}:
        return {"ok": False, "error": "invalid_direction", "allowed": ["out", "in", "both"]}
    rel_filter = ""
    if rel:
        # Relationship type must be a valid identifier-ish token.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rel):
            return {"ok": False, "error": "invalid_rel"}
        rel_filter = f":{rel}"

    if direction == "out":
        q = f"MATCH (a {{id: $id}})-[r{rel_filter}]->(b) RETURN type(r) AS rel, b.id AS id, labels(b) AS labels LIMIT $limit"
    elif direction == "in":
        q = f"MATCH (a {{id: $id}})<-[r{rel_filter}]-(b) RETURN type(r) AS rel, b.id AS id, labels(b) AS labels LIMIT $limit"
    else:
        q = f"MATCH (a {{id: $id}})-[r{rel_filter}]-(b) RETURN type(r) AS rel, b.id AS id, labels(b) AS labels LIMIT $limit"

    g = _connect()
    res = g.query(q, {"id": node_id, "limit": max(1, min(int(limit or 50), 500))})
    return {"ok": True, "graph": _graph_name(), "result": _coerce_result(res, limit=limit)}


@mcp.tool()
def shortest_path(
    src_id: str,
    dst_id: str,
    rel: str = "CALLS",
    max_hops: int = 10,
) -> dict[str, Any]:
    """Compute a shortest path between two nodes by id (within a single relationship type)."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rel or ""):
        return {"ok": False, "error": "invalid_rel"}
    max_hops = max(1, min(int(max_hops or 10), 50))
    q = (
        f"MATCH (a {{id: $src}}), (b {{id: $dst}}) "
        f"MATCH p=shortestPath((a)-[:{rel}*..{max_hops}]->(b)) "
        "RETURN [n IN nodes(p) | n.id] AS nodes, [r IN relationships(p) | type(r)] AS rels "
        "LIMIT 1"
    )
    g = _connect()
    res = g.query(q, {"src": src_id, "dst": dst_id})
    return {"ok": True, "graph": _graph_name(), "result": _coerce_result(res, limit=1)}


@mcp.tool()
def top_out_degree(
    rel: str = "CALLS",
    label: str = "Function",
    top_k: int = 25,
) -> dict[str, Any]:
    """Simple graph-analytics: top nodes by out-degree for a relationship type."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rel or ""):
        return {"ok": False, "error": "invalid_rel"}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label or ""):
        return {"ok": False, "error": "invalid_label"}
    top_k = max(1, min(int(top_k or 25), 200))
    q = (
        f"MATCH (n:{label})-[r:{rel}]->() "
        "RETURN n.id AS id, count(r) AS out_degree "
        "ORDER BY out_degree DESC "
        "LIMIT $k"
    )
    g = _connect()
    res = g.query(q, {"k": top_k})
    return {"ok": True, "graph": _graph_name(), "result": _coerce_result(res, limit=top_k)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_SERVER_PORT", "8001")))
    parser.add_argument("--host", type=str, default=os.getenv("MCP_SERVER_HOST", "0.0.0.0"))
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    app = mcp.sse_app()

    def root(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    app.add_route("/", root, methods=["GET"])
    app.add_route("/health", health, methods=["GET"])

    uvicorn.run(app, host=args.host, port=args.port)

