import argparse
import json
from typing import Any

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import AdaptiveCrawler, AdaptiveConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

mcp = FastMCP("mcp-crawl4ai")


class CrawlResult(BaseModel):
    url: str
    title: str | None = None
    markdown: str | None = None
    text: str | None = None
    links: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@mcp.tool()
async def crawl_url(
    url: str,
    query: str | None = None,
    include_markdown: bool = True,
    include_text: bool = True,
    use_fit_markdown: bool = True,
    max_chars: int = 12000,
    adaptive: bool = True,
    adaptive_max_pages: int = 12,
    adaptive_top_k: int = 5,
) -> str:
    """Fetch a single URL using Crawl4AI and return extracted content.

    Returns a JSON string with fields: url, title, markdown, text, links, metadata.
    """

    if not url:
        raise ValueError("url is required")

    if AsyncWebCrawler is None:
        return json.dumps(
            {
                "error": "crawl4ai is not available in this runtime. Ensure dependencies are installed.",
                "url": url,
            }
        )

    # Tighten optional-import types for static checkers. At runtime, if crawl4ai is
    # present, these symbols are classes; otherwise we returned above already.
    assert CrawlerRunConfig is not None
    assert DefaultMarkdownGenerator is not None
    assert BM25ContentFilter is not None
    assert PruningContentFilter is not None

    # Build a markdown generator that outputs "fit_markdown" to reduce boilerplate
    # and keep agent context small.
    #
    # Docs: https://docs.crawl4ai.com/core/fit-markdown/
    try:
        if query:
            content_filter = BM25ContentFilter(user_query=query, bm25_threshold=1.2)
        else:
            content_filter = PruningContentFilter(
                threshold=0.45,
                threshold_type="dynamic",
                min_word_threshold=8,
            )
        md_generator = DefaultMarkdownGenerator(content_filter=content_filter)
        run_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            excluded_tags=["nav", "footer", "header", "aside"],
            word_count_threshold=8,
        )
    except Exception as e:  # pragma: no cover
        return json.dumps({"error": f"Failed to build Crawl4AI config: {e}", "url": url})

    async with AsyncWebCrawler(verbose=False) as crawler:
        # Adaptive crawling is the best "research" default when a query is provided:
        # follow relevant links and stop when information sufficiency is reached.
        #
        # If adaptive crawler isn't available at runtime, we gracefully fall back to single-page `arun`.
        use_adaptive = bool(adaptive and query and AdaptiveCrawler is not None and AdaptiveConfig is not None)
        if use_adaptive:
            try:
                assert query is not None
                assert AdaptiveConfig is not None
                assert AdaptiveCrawler is not None
                adaptive_config = AdaptiveConfig(
                    strategy="statistical",
                    confidence_threshold=0.75,
                    max_pages=adaptive_max_pages,
                    top_k_links=5,
                    min_gain_threshold=0.08,
                )
                adaptive_crawler = AdaptiveCrawler(crawler, adaptive_config)
                await adaptive_crawler.digest(start_url=url, query=query)
                relevant_pages = adaptive_crawler.get_relevant_content(top_k=adaptive_top_k) or []

                # Flatten relevant pages into a compact payload
                links = []
                excerpts = []
                for p in relevant_pages:
                    p_url = (p or {}).get("url")
                    if p_url:
                        links.append(p_url)
                    score = (p or {}).get("score")
                    content = (p or {}).get("content") or ""
                    if isinstance(content, str) and max_chars > 0 and len(content) > max_chars:
                        content = content[:max_chars] + "\n\n[...truncated...]"
                    excerpts.append(
                        {
                            "url": p_url,
                            "score": score,
                            "content": content,
                        }
                    )

                # Return in the standard CrawlResult envelope, keeping heavy data under metadata.
                payload = CrawlResult(
                    url=url,
                    title=None,
                    markdown=json.dumps(excerpts, default=str) if include_markdown else None,
                    text=None,
                    links=links,
                    metadata={
                        "mode": "adaptive",
                        "query": query,
                        "adaptive_max_pages": adaptive_max_pages,
                        "adaptive_top_k": adaptive_top_k,
                        "returned_pages": len(excerpts),
                        "max_chars": max_chars,
                    },
                )
                return payload.model_dump_json()
            except Exception as e:
                # Fall back to single-page crawl if adaptive fails for any reason.
                pass

        # crawl4ai types can vary across versions; treat the result as dynamic.
        result: Any = await crawler.arun(url=url, config=run_config)

    # Resolve markdown / fit_markdown defensively (crawl4ai result schema can vary by version)
    raw_markdown = None
    fit_markdown = None
    try:
        md = getattr(result, "markdown", None)
        raw_markdown = getattr(md, "raw_markdown", None) if md else None
        fit_markdown = getattr(md, "fit_markdown", None) if md else None
    except Exception:
        raw_markdown = None
        fit_markdown = None

    chosen_markdown = None
    if include_markdown:
        chosen_markdown = (fit_markdown if use_fit_markdown else raw_markdown) or raw_markdown or fit_markdown
        if isinstance(chosen_markdown, str) and max_chars > 0 and len(chosen_markdown) > max_chars:
            chosen_markdown = chosen_markdown[:max_chars] + "\n\n[...truncated...]"

    chosen_text = None
    if include_text:
        chosen_text = getattr(result, "text", None)
        if isinstance(chosen_text, str) and max_chars > 0 and len(chosen_text) > max_chars:
            chosen_text = chosen_text[:max_chars] + "\n\n[...truncated...]"

    links: list[str] = []
    try:
        if getattr(result, "links", None):
            if isinstance(result.links, dict):
                for v in result.links.values():
                    if isinstance(v, list):
                        for item in v:
                            href = item.get("href") if isinstance(item, dict) else None
                            if href:
                                links.append(href)
            elif isinstance(result.links, list):
                for item in result.links:
                    href = item.get("href") if isinstance(item, dict) else None
                    if href:
                        links.append(href)
    except Exception:
        links = []

    payload = CrawlResult(
        url=url,
        title=getattr(result, "title", None),
        markdown=chosen_markdown,
        text=chosen_text,
        links=links,
        metadata={
            "mode": "single",
            "success": getattr(result, "success", None),
            "status": getattr(result, "status", None),
            "content_type": getattr(result, "content_type", None),
            "used_fit_markdown": bool(use_fit_markdown),
            "query": query or "",
            "raw_markdown_len": len(raw_markdown) if isinstance(raw_markdown, str) else None,
            "fit_markdown_len": len(fit_markdown) if isinstance(fit_markdown, str) else None,
            "returned_markdown_len": len(chosen_markdown) if isinstance(chosen_markdown, str) else None,
            "returned_text_len": len(chosen_text) if isinstance(chosen_text, str) else None,
            "max_chars": max_chars,
        },
    )

    return payload.model_dump_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    app = mcp.sse_app()

    def root(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    # `mcp.sse_app()` is typed as a Starlette app; Starlette doesn't expose `.get`.
    app.add_route("/", root, methods=["GET"])
    app.add_route("/health", health, methods=["GET"])

    uvicorn.run(app, host=args.host, port=args.port)
