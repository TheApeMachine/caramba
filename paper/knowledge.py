"""Knowledge store for academic papers using Deep Lake and ColBERT.

This module provides a persistent, searchable knowledge base for papers
discovered during research. It uses:

- Deep Lake: Vector database for storing papers and embeddings
- Docling: PDF parsing for extracting text, tables, and figures
- ColBERT: Late-interaction embeddings for precise academic search

The knowledge store builds up over time, reducing API calls and enabling
rich semantic search over previously discovered papers.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from console import logger

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    import deeplake
    import torch
    from docling.document_converter import DocumentConverter


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Paper:
    """A parsed academic paper with embeddings."""

    paper_id: str
    title: str
    authors: list[str]
    year: int
    venue: str
    abstract: str | None = None
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None

    # Parsed content
    full_text: str | None = None
    sections: dict[str, str] = field(default_factory=dict)
    figures: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)

    # Embeddings (stored as lists for serialization)
    passage_embeddings: list[list[float]] = field(default_factory=list)
    passage_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "abstract": self.abstract,
            "url": self.url,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "full_text": self.full_text,
            "sections": self.sections,
            "figures": self.figures,
            "tables": self.tables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        """Create from dictionary."""
        return cls(
            paper_id=data["paper_id"],
            title=data["title"],
            authors=data.get("authors", []),
            year=data.get("year", 0),
            venue=data.get("venue", ""),
            abstract=data.get("abstract"),
            url=data.get("url"),
            doi=data.get("doi"),
            arxiv_id=data.get("arxiv_id"),
            full_text=data.get("full_text"),
            sections=data.get("sections", {}),
            figures=data.get("figures", []),
            tables=data.get("tables", []),
        )


@dataclass
class SearchResult:
    """A search result from the knowledge store."""

    paper: Paper
    score: float
    matched_passage: str | None = None


# ============================================================================
# PDF Parser using Docling
# ============================================================================


class PaperParser:
    """Parse academic papers using Docling.

    Extracts structured content including:
    - Full text with section boundaries
    - Tables (as structured data)
    - Figures (as images with captions)
    - References
    """

    def __init__(self) -> None:
        self._converter: "DocumentConverter | None" = None

    @property
    def converter(self) -> "DocumentConverter":
        """Lazy-load the Docling converter."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter

                self._converter = DocumentConverter()
                logger.info("Docling document converter initialized")
            except ImportError:
                raise ImportError(
                    "Docling not installed. Run: pip install docling"
                )
        return self._converter

    def parse_pdf(self, pdf_path: Path | str) -> dict[str, Any]:
        """Parse a PDF file and extract structured content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary with extracted content.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")

        result = self.converter.convert(str(pdf_path))
        doc = result.document

        # Extract sections
        sections: dict[str, str] = {}
        current_section = "preamble"
        current_text: list[str] = []

        for item, _level in doc.iterate_items():
            label = getattr(item, "label", None)
            text = getattr(item, "text", None)
            if label == "section_header":
                # Save previous section
                if current_text:
                    sections[current_section] = "\n".join(current_text)
                current_section = str(text) if text else "unnamed"
                current_text = []
            elif text:
                current_text.append(str(text))

        # Save last section
        if current_text:
            sections[current_section] = "\n".join(current_text)

        # Extract tables
        tables: list[dict[str, Any]] = []
        for table in doc.tables:
            tables.append({
                "caption": getattr(table, "caption", ""),
                "data": table.export_to_dataframe().to_dict() if hasattr(table, "export_to_dataframe") else {},
            })

        # Extract figures
        figures: list[dict[str, Any]] = []
        for figure in doc.pictures:
            figures.append({
                "caption": getattr(figure, "caption", ""),
                "page": getattr(figure, "page_no", 0),
            })

        # Get full text
        full_text = doc.export_to_markdown()

        return {
            "full_text": full_text,
            "sections": sections,
            "tables": tables,
            "figures": figures,
        }

    def parse_from_url(self, url: str) -> dict[str, Any]:
        """Download and parse a PDF from URL.

        Args:
            url: URL to the PDF (e.g., arXiv PDF link).

        Returns:
            Dictionary with extracted content.
        """
        logger.info(f"Downloading PDF from: {url}")

        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Caramba/1.0 (research platform)")
            with urllib.request.urlopen(req, timeout=60) as response:
                tmp.write(response.read())
            tmp_path = Path(tmp.name)

        try:
            return self.parse_pdf(tmp_path)
        finally:
            tmp_path.unlink()  # Clean up temp file


# ============================================================================
# ColBERT Embedder
# ============================================================================


class ColBERTEmbedder:
    """Generate ColBERT embeddings for late-interaction search.

    ColBERT (Contextualized Late Interaction over BERT) generates
    per-token embeddings that enable fine-grained matching between
    queries and documents using MaxSim.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None

    def _load_model(self) -> None:
        """Lazy-load the ColBERT model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            logger.info(f"Loading ColBERT model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            elif torch.backends.mps.is_available():
                self._model = self._model.to("mps")

            self._model.eval()
            logger.success("ColBERT model loaded")

        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

    def embed_passages(
        self,
        passages: list[str],
        batch_size: int = 8,
    ) -> list[list[list[float]]]:
        """Generate ColBERT embeddings for passages.

        Args:
            passages: List of text passages to embed.
            batch_size: Batch size for processing.

        Returns:
            List of embeddings, each is a 2D array (tokens x dim).
        """
        self._load_model()
        import torch

        all_embeddings: list[list[list[float]]] = []

        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to same device as model
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state

            # Convert to list format
            for emb in embeddings:
                all_embeddings.append(emb.cpu().tolist())

        return all_embeddings

    def embed_query(self, query: str) -> list[list[float]]:
        """Generate ColBERT embedding for a query.

        Args:
            query: The search query.

        Returns:
            2D embedding array (tokens x dim).
        """
        return self.embed_passages([query])[0]

    @staticmethod
    def maxsim(
        query_emb: list[list[float]],
        doc_emb: list[list[float]],
    ) -> float:
        """Compute MaxSim score between query and document embeddings.

        For each query token, find the max similarity with any doc token,
        then sum across all query tokens.

        Args:
            query_emb: Query embedding (tokens x dim).
            doc_emb: Document embedding (tokens x dim).

        Returns:
            MaxSim similarity score.
        """
        q = np.array(query_emb)
        d = np.array(doc_emb)

        # Compute cosine similarity matrix
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        sim_matrix = q_norm @ d_norm.T

        # MaxSim: for each query token, take max similarity with any doc token
        max_sims = sim_matrix.max(axis=1)
        return float(max_sims.sum())


# ============================================================================
# Knowledge Store
# ============================================================================


class KnowledgeStore:
    """Persistent knowledge base for academic papers.

    Uses Deep Lake to store papers with ColBERT embeddings for
    efficient semantic search.
    """

    def __init__(
        self,
        store_path: Path | str = "~/.caramba/knowledge",
        use_colbert: bool = True,
    ) -> None:
        """Initialize the knowledge store.

        Args:
            store_path: Path to the Deep Lake dataset.
            use_colbert: Whether to use ColBERT embeddings.
        """
        self.store_path = Path(store_path).expanduser()
        self.use_colbert = use_colbert

        self._ds: "deeplake.Dataset | None" = None
        self._parser: PaperParser | None = None
        self._embedder: ColBERTEmbedder | None = None

    @property
    def parser(self) -> PaperParser:
        """Lazy-load the paper parser."""
        if self._parser is None:
            self._parser = PaperParser()
        return self._parser

    @property
    def embedder(self) -> ColBERTEmbedder:
        """Lazy-load the ColBERT embedder."""
        if self._embedder is None:
            self._embedder = ColBERTEmbedder()
        return self._embedder

    def _get_dataset(self) -> "deeplake.Dataset":
        """Get or create the Deep Lake dataset."""
        if self._ds is not None:
            return self._ds

        try:
            import deeplake
            from deeplake import types  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                "deeplake not installed. Run: pip install deeplake"
            )

        self.store_path.mkdir(parents=True, exist_ok=True)
        dataset_path = f"file://{self.store_path}/papers"

        try:
            # Try to open existing dataset
            ds = deeplake.open(dataset_path)  # type: ignore[attr-defined]
            self._ds = ds
            logger.info(f"Opened knowledge store with {len(ds)} papers")
        except Exception:
            # Create new dataset
            logger.info("Creating new knowledge store...")
            ds = deeplake.create(dataset_path)  # type: ignore[attr-defined]
            self._ds = ds

            # Add columns for paper metadata
            ds.add_column("paper_id", types.Text())
            ds.add_column("title", types.Text())
            ds.add_column("authors", types.Text())  # JSON list
            ds.add_column("year", types.Int32())
            ds.add_column("venue", types.Text())
            ds.add_column("abstract", types.Text())
            ds.add_column("url", types.Text())
            ds.add_column("doi", types.Text())
            ds.add_column("arxiv_id", types.Text())
            ds.add_column("full_text", types.Text())
            ds.add_column("sections", types.Text())  # JSON dict
            ds.add_column("metadata", types.Text())  # JSON for extras

            # Add columns for embeddings (ColBERT produces 2D arrays)
            if self.use_colbert:
                ds.add_column(
                    "passage_embeddings",
                    types.Array(types.Float32(), dimensions=2),
                )
                ds.add_column("passage_texts", types.Text())  # JSON list

            ds.commit()
            logger.success("Knowledge store created")

        return self._ds  # type: ignore[return-value]

    def add_paper(
        self,
        paper: Paper,
        parse_pdf: bool = False,
        pdf_url: str | None = None,
    ) -> None:
        """Add a paper to the knowledge store.

        Args:
            paper: Paper object with metadata.
            parse_pdf: Whether to download and parse the PDF.
            pdf_url: URL to the PDF (auto-detected from arXiv if not provided).
        """
        ds = self._get_dataset()

        # Check if paper already exists
        if self._paper_exists(paper.paper_id):
            logger.info(f"Paper already in store: {paper.title[:50]}...")
            return

        # Parse PDF if requested
        if parse_pdf:
            url = pdf_url
            if url is None and paper.arxiv_id:
                url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

            if url:
                try:
                    parsed = self.parser.parse_from_url(url)
                    paper.full_text = parsed.get("full_text")
                    paper.sections = parsed.get("sections", {})
                    paper.figures = parsed.get("figures", [])
                    paper.tables = parsed.get("tables", [])
                except Exception as e:
                    logger.warning(f"Failed to parse PDF: {e}")

        # Generate embeddings if using ColBERT
        # Each passage gets a 2D embedding (tokens x dim)
        passage_embeddings: list[list[list[float]]] = []
        passage_texts: list[str] = []

        if self.use_colbert and paper.full_text:
            # Split into passages (simple chunking by paragraphs)
            passages = self._chunk_text(paper.full_text)
            passage_texts = passages

            try:
                embeddings = self.embedder.embed_passages(passages)
                passage_embeddings = embeddings
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")

        # Add to dataset
        ds.append({
            "paper_id": [paper.paper_id],
            "title": [paper.title],
            "authors": [json.dumps(paper.authors)],
            "year": [paper.year],
            "venue": [paper.venue],
            "abstract": [paper.abstract or ""],
            "url": [paper.url or ""],
            "doi": [paper.doi or ""],
            "arxiv_id": [paper.arxiv_id or ""],
            "full_text": [paper.full_text or ""],
            "sections": [json.dumps(paper.sections)],
            "metadata": [json.dumps({})],
            **({"passage_embeddings": passage_embeddings,
                "passage_texts": [json.dumps(passage_texts)]} if self.use_colbert else {}),
        })
        ds.commit()  # type: ignore[union-attr]

        logger.success(f"Added paper: {paper.title[:50]}...")

    def _paper_exists(self, paper_id: str) -> bool:
        """Check if a paper already exists in the store."""
        ds = self._get_dataset()
        if len(ds) == 0:
            return False

        # Simple linear scan for now (Deep Lake will optimize this)
        for row in ds:
            if row["paper_id"] == paper_id:  # type: ignore[index]
                return True
        return False

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk.
            chunk_size: Target chunk size in characters.
            overlap: Overlap between chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_length + len(para) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))

                # Start new chunk with overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_length = len(overlap_text) + len(para)
            else:
                current_chunk.append(para)
                current_length += len(para)

        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_colbert: bool | None = None,
    ) -> list[SearchResult]:
        """Search the knowledge store.

        Args:
            query: Search query.
            top_k: Number of results to return.
            use_colbert: Whether to use ColBERT search (default: self.use_colbert).

        Returns:
            List of search results with scores.
        """
        ds = self._get_dataset()

        if len(ds) == 0:
            return []

        use_colbert = use_colbert if use_colbert is not None else self.use_colbert

        if use_colbert:
            return self._colbert_search(query, top_k)
        else:
            return self._text_search(query, top_k)

    def _colbert_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Perform ColBERT late-interaction search."""
        ds = self._get_dataset()

        # Get query embedding
        query_emb = self.embedder.embed_query(query)

        results: list[tuple[int, float, str | None]] = []

        for idx in range(len(ds)):
            row = ds[idx]  # type: ignore[index]

            # Get passage embeddings
            if "passage_embeddings" not in row:  # type: ignore[operator]
                continue

            passage_embs = row["passage_embeddings"]  # type: ignore[index]
            if passage_embs is None or len(passage_embs) == 0:
                continue

            # Compute MaxSim for each passage, take the best
            best_score = 0.0
            best_passage: str | None = None

            passage_texts = json.loads(row.get("passage_texts", "[]"))  # type: ignore[union-attr]

            for i, doc_emb in enumerate(passage_embs):
                score = ColBERTEmbedder.maxsim(query_emb, list(doc_emb))  # type: ignore[arg-type]
                if score > best_score:
                    best_score = score
                    best_passage = passage_texts[i] if i < len(passage_texts) else None

            results.append((idx, best_score, best_passage))

        # Sort by score and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        # Convert to SearchResult objects
        search_results: list[SearchResult] = []
        for idx, score, passage in results:
            row = ds[idx]  # type: ignore[index]
            paper = Paper(
                paper_id=row["paper_id"],  # type: ignore[index]
                title=row["title"],  # type: ignore[index]
                authors=json.loads(row.get("authors", "[]")),  # type: ignore[union-attr]
                year=row.get("year", 0),  # type: ignore[union-attr]
                venue=row.get("venue", ""),  # type: ignore[union-attr]
                abstract=row.get("abstract"),  # type: ignore[union-attr]
                url=row.get("url"),  # type: ignore[union-attr]
                doi=row.get("doi"),  # type: ignore[union-attr]
                arxiv_id=row.get("arxiv_id"),  # type: ignore[union-attr]
                full_text=row.get("full_text"),  # type: ignore[union-attr]
                sections=json.loads(row.get("sections", "{}")),  # type: ignore[union-attr]
            )
            search_results.append(SearchResult(
                paper=paper,
                score=score,
                matched_passage=passage,
            ))

        return search_results

    def _text_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Perform simple text search (fallback when ColBERT not available)."""
        ds = self._get_dataset()

        query_terms = set(query.lower().split())
        results: list[tuple[int, float]] = []

        for idx in range(len(ds)):
            row = ds[idx]  # type: ignore[index]
            text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()  # type: ignore[union-attr]
            text_terms = set(text.split())

            # Simple term overlap scoring
            overlap = len(query_terms & text_terms)
            score = overlap / len(query_terms) if query_terms else 0

            if score > 0:
                results.append((idx, score))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        search_results: list[SearchResult] = []
        for idx, score in results:
            row = ds[idx]  # type: ignore[index]
            paper = Paper(
                paper_id=row["paper_id"],  # type: ignore[index]
                title=row["title"],  # type: ignore[index]
                authors=json.loads(row.get("authors", "[]")),  # type: ignore[union-attr]
                year=row.get("year", 0),  # type: ignore[union-attr]
                venue=row.get("venue", ""),  # type: ignore[union-attr]
                abstract=row.get("abstract"),  # type: ignore[union-attr]
                url=row.get("url"),  # type: ignore[union-attr]
            )
            search_results.append(SearchResult(paper=paper, score=score))

        return search_results

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge store."""
        ds = self._get_dataset()
        return {
            "total_papers": len(ds),
            "store_path": str(self.store_path),
            "use_colbert": self.use_colbert,
        }


# ============================================================================
# Helper Functions
# ============================================================================


def generate_paper_id(title: str, authors: list[str], year: int) -> str:
    """Generate a unique paper ID from metadata."""
    key = f"{title.lower()}|{','.join(a.lower() for a in authors[:3])}|{year}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


# ============================================================================
# Module-level singleton
# ============================================================================

_knowledge_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    """Get or create the singleton knowledge store."""
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = KnowledgeStore()
    return _knowledge_store
