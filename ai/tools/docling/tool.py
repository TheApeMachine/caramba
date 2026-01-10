"""Docling document conversion tool

Exposes a small MCP server for converting PDFs (and other supported formats) into
agent-friendly outputs like Markdown or JSON. This is needed because PDFs are often
the primary format for papers and reports, and agents should work with structured
text rather than raw binary bytes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from docling.document_converter import DocumentConverter


mcp = FastMCP("Docling Tool", json_response=True)
mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)


def _allowed_roots() -> list[Path]:
    """Allowed roots for local file access

    Restricts conversions to a small set of mounted directories so the server cannot
    read arbitrary container paths. This is the same safety boundary used by our
    filesystem MCP tool.
    """
    raw = os.getenv("DOCLING_ALLOWED_ROOTS", "/app/artifacts,/app/config,/app/docs,/app/research").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    roots: list[Path] = []
    for p in parts:
        roots.append(Path(p).resolve())
    return roots


_ROOTS = _allowed_roots()
_BASE_DIR = Path("/app").resolve()


def _resolve_and_check(path: str) -> str:
    """Path resolution and root enforcement

    Ensures a provided path resolves under an allowlisted root. This prevents agents
    from exfiltrating arbitrary files from the container filesystem.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    p = Path(path.strip())
    if not p.is_absolute():
        p = _BASE_DIR / p
    rp = p.resolve()
    for root in _ROOTS:
        if rp == root:
            return str(rp)
        if root.is_dir():
            rp.relative_to(root)
            return str(rp)
    allowed = ", ".join(str(r) for r in _ROOTS) if _ROOTS else "(none)"
    raise PermissionError(f"Access denied. Path not under allowed roots: {allowed}")


def _truncate(text: str, *, max_chars: int) -> str:
    """Output truncation helper

    Prevents overly large responses from being returned into agent context. This keeps
    the tool safe to call in long-running chats and aligns with other content tools.
    """
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated...]"


@mcp.tool()
def convert_document(
    source: str,
    *,
    output: str = "markdown",
    max_chars: int = 12000,
) -> str:
    """Convert a document into agent-friendly text

    Uses Docling to parse a local file path (under an allowed root) or a URL and
    returns either Markdown or JSON (as a string). This is the primary interface
    agents will use to read PDFs.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("output must be a non-empty string")
    output = output.strip().lower()
    if output not in {"markdown", "md", "json", "dict"}:
        raise ValueError("output must be one of: markdown, md, json, dict")
    if not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("max_chars must be an int >= 0")

    src = source.strip()
    if src.startswith(("http://", "https://")):
        resolved_source = src
    else:
        resolved_source = _resolve_and_check(src)

    converter = DocumentConverter()
    result = converter.convert(resolved_source)
    doc = result.document

    if output in {"markdown", "md"}:
        markdown = doc.export_to_markdown()
        payload = {
            "ok": True,
            "source": resolved_source,
            "format": "markdown",
            "content": _truncate(markdown, max_chars=max_chars),
        }
        return json.dumps(payload, ensure_ascii=False)

    as_dict = doc.export_to_dict()
    payload = {
        "ok": True,
        "source": resolved_source,
        "format": "json",
        "content": as_dict,
    }
    raw = json.dumps(payload, ensure_ascii=False)
    return _truncate(raw, max_chars=max_chars)


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

