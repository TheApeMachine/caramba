"""AI-assisted paper drafting and review for experiments.

This module provides automated paper generation and review from experiment
results using OpenAI's Agent SDK. It can:

- Create new LaTeX papers from experiment manifests and results
- Update existing drafts with new experiment data
- Search and cite relevant literature (with local caching via Deep Lake)
- Generate figures and tables from artifacts
- Review papers and identify weaknesses
- Propose and generate new experiments to strengthen papers
- Run autonomous research loops (write → review → experiment → repeat)
- Build a persistent knowledge base of discovered papers using ColBERT embeddings
"""
from paper.drafter import PaperDrafter
from paper.reviewer import PaperReviewer
from paper.research_loop import ResearchLoop, ResearchLoopConfig
from paper.research_team import ResearchTeam, DiscussionTranscript, ResearchProposal
from paper.review import ReviewConfig, ReviewResult

# Knowledge store is optional (requires deeplake, docling, transformers)
try:
    from paper.knowledge import (
        KnowledgeStore,
        Paper,
        PaperParser,
        ColBERTEmbedder,
        get_knowledge_store,
    )
    _KNOWLEDGE_AVAILABLE = True
except ImportError:
    _KNOWLEDGE_AVAILABLE = False

__all__ = [
    "PaperDrafter",
    "PaperReviewer",
    "ResearchLoop",
    "ResearchLoopConfig",
    "ResearchTeam",
    "DiscussionTranscript",
    "ResearchProposal",
    "ReviewConfig",
    "ReviewResult",
]

# Add knowledge exports if available
if _KNOWLEDGE_AVAILABLE:
    __all__.extend([
        "KnowledgeStore",
        "Paper",
        "PaperParser",
        "ColBERTEmbedder",
        "get_knowledge_store",
    ])
