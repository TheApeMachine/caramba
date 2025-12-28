"""MCP server for research paper writing tools.

Run this as a standalone server:
    python -m agent.tools.paper

Then connect to it via MCPServerStreamableHttp at http://localhost:8001/mcp
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Paper Tools", json_response=True)

# Get paper path from environment or use default
def get_paper_path() -> Path:
    """Get the paper path from environment variable or default."""
    env_path = os.getenv("PAPER_PATH")
    if env_path:
        path = Path(env_path)
    else:
        # Default to artifacts directory structure
        path = Path(__file__).parent.parent.parent / "artifacts" / "llama32_1b_dba_paper"
    # Ensure directory exists
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_paper_file(paper_path: Path) -> Path | None:
    """Get the latest paper file path."""
    versions = [
        int(path.stem.split("_")[-1])
        for path in paper_path.glob("paper_*.tex")
    ]
    if not versions:
        return None
    latest_version = max(versions)
    return paper_path / f"paper_{latest_version}.tex"


def get_references_file(paper_path: Path) -> Path:
    """Get the references.bib file path."""
    return paper_path / "references.bib"


def parse_latex_sections(content: str) -> dict[str, str]:
    """Parse LaTeX document into sections.

    Returns a dict mapping section names to their content.
    """
    sections: dict[str, str] = {}

    # Pattern to match \section{name} or \subsection{name} etc.
    section_pattern = r'\\(?:section|subsection|subsubsection)\*?\{([^}]+)\}'

    lines = content.split('\n')
    current_section = "preamble"
    current_content: list[str] = []

    for line in lines:
        match = re.search(section_pattern, line)
        if match:
            # Save previous section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            # Start new section
            current_section = match.group(1).lower().replace(' ', '_')
            current_content = [line]
        else:
            current_content.append(line)

    # Save last section
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections


def replace_section(content: str, section_name: str, new_content: str) -> str:
    """Replace or add a section in LaTeX content."""
    # Normalize section name
    section_name_lower = section_name.lower().replace(' ', '_')

    # Try to find existing section
    section_pattern = rf'\\(?:section|subsection|subsubsection)\*?\{{{re.escape(section_name)}\}}'

    if re.search(section_pattern, content, re.IGNORECASE):
        # Replace existing section
        pattern = rf'(\\(?:section|subsection|subsubsection)\*?\{{{re.escape(section_name)}\}}.*?)(?=\\(?:section|subsection|subsubsection|\\end|$)'
        replacement = rf'\1\n{new_content}'
        return re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
    else:
        # Add new section before \end{document} or at end
        if '\\end{document}' in content:
            content = content.replace('\\end{document}', f'\\section{{{section_name}}}\n{new_content}\n\n\\end{{document}}')
        else:
            content += f'\n\\section{{{section_name}}}\n{new_content}\n'
        return content


@mcp.tool()
def read_tex_file() -> str:
    """Read the current paper.tex file.

    Returns the full LaTeX content of the latest paper version.
    """
    paper_path = get_paper_path()
    paper_file = get_latest_paper_file(paper_path)

    if paper_file is None or not paper_file.exists():
        return "No paper found. Use write_tex_file to create one."

    with open(paper_file, "r") as f:
        return f.read()


@mcp.tool()
def write_tex_file(content: str) -> str:
    """Write the complete paper LaTeX file.

    Args:
        content: The complete LaTeX document content.

    Creates a new versioned file (paper_N.tex) preserving previous versions.
    """
    paper_path = get_paper_path()
    latest_file = get_latest_paper_file(paper_path)

    if latest_file:
        latest_version = int(latest_file.stem.split("_")[-1])
        version = latest_version + 1
    else:
        version = 1

    paper_file = paper_path / f"paper_{version}.tex"
    with open(paper_file, "w") as f:
        f.write(content)

    return f"Paper written successfully to {paper_file.name}"


@mcp.tool()
def update_section(section_name: str, content: str) -> str:
    """Update a specific section in the paper.

    Args:
        section_name: Name of the section to update (e.g., "Introduction", "Methodology")
        content: The new content for the section (LaTeX formatted)

    If the section doesn't exist, it will be created.
    """
    paper_path = get_paper_path()
    paper_file = get_latest_paper_file(paper_path)

    if paper_file is None or not paper_file.exists():
        return "No paper found. Use write_tex_file to create one first."

    with open(paper_file, "r") as f:
        latex_content = f.read()

    updated_content = replace_section(latex_content, section_name, content)

    # Write updated version
    latest_version = int(paper_file.stem.split("_")[-1])
    new_version = latest_version + 1
    new_file = paper_path / f"paper_{new_version}.tex"

    with open(new_file, "w") as f:
        f.write(updated_content)

    return f"Section '{section_name}' updated successfully. New version: {new_file.name}"


@mcp.tool()
def get_section(section_name: str) -> str:
    """Get the content of a specific section.

    Args:
        section_name: Name of the section to retrieve
    """
    paper_path = get_paper_path()
    paper_file = get_latest_paper_file(paper_path)

    if paper_file is None or not paper_file.exists():
        return "No paper found."

    with open(paper_file, "r") as f:
        content = f.read()

    sections = parse_latex_sections(content)
    section_key = section_name.lower().replace(' ', '_')

    if section_key in sections:
        return sections[section_key]
    else:
        available = ', '.join(sections.keys())
        return f"Section '{section_name}' not found. Available sections: {available}"


@mcp.tool()
def add_citation(bibtex_entry: str) -> str:
    """Add a BibTeX citation entry to references.bib.

    Args:
        bibtex_entry: The complete BibTeX entry (e.g., @article{key2024, title={...}, ...})

    The entry will be appended to references.bib. If the file doesn't exist, it will be created.
    """
    paper_path = get_paper_path()
    ref_file = get_references_file(paper_path)

    # Read existing entries
    existing_entries = ""
    if ref_file.exists():
        with open(ref_file, "r") as f:
            existing_entries = f.read()

    # Check if entry already exists (by key)
    entry_key_match = re.search(r'@\w+\{([^,]+),', bibtex_entry)
    if entry_key_match:
        entry_key = entry_key_match.group(1)
        if entry_key in existing_entries:
            return f"Citation with key '{entry_key}' already exists in references.bib"

    # Append new entry
    with open(ref_file, "a") as f:
        if existing_entries and not existing_entries.endswith('\n'):
            f.write('\n')
        f.write(bibtex_entry)
        f.write('\n\n')

    return f"Citation added successfully to {ref_file.name}"


@mcp.tool()
def search_arxiv(query: str, max_results: int = 10) -> str:
    """Search arXiv for research papers.

    Args:
        query: Search query (keywords, title, author, etc.)
        max_results: Maximum number of results to return (default: 10)

    Returns a formatted list of papers with titles, authors, abstracts, and arXiv IDs.
    """
    try:
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        # arXiv API endpoint
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        with urllib.request.urlopen(url) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        results = []
        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title"

            author_elems = entry.findall('atom:author', ns)
            authors = []
            for author_elem in author_elems:
                name_elem = author_elem.find('atom:name', ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)

            summary_elem = entry.find('atom:summary', ns)
            summary = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else "No abstract"

            id_elem = entry.find('atom:id', ns)
            arxiv_id = id_elem.text.split('/')[-1] if id_elem is not None and id_elem.text else "Unknown"

            published_elem = entry.find('atom:published', ns)
            published = published_elem.text[:10] if published_elem is not None and published_elem.text else "Unknown"

            results.append({
                'title': title,
                'authors': ', '.join(authors) if authors else "Unknown authors",
                'arxiv_id': arxiv_id,
                'published': published,
                'abstract': summary[:500] + "..." if len(summary) > 500 else summary
            })

        if not results:
            return f"No papers found for query: {query}"

        output = f"Found {len(results)} papers for '{query}':\n\n"
        for i, paper in enumerate(results, 1):
            output += f"{i}. {paper['title']}\n"
            output += f"   Authors: {paper['authors']}\n"
            output += f"   arXiv: {paper['arxiv_id']} ({paper['published']})\n"
            output += f"   Abstract: {paper['abstract']}\n\n"

        return output

    except Exception as e:
        return f"Error searching arXiv: {str(e)}"


@mcp.tool()
def search_semantic_scholar(query: str, max_results: int = 10) -> str:
    """Search Semantic Scholar for research papers.

    Args:
        query: Search query (keywords, title, author, etc.)
        max_results: Maximum number of results to return (default: 10)

    Note: This requires a Semantic Scholar API key. Returns a formatted list of papers.
    """
    try:
        import urllib.parse
        import urllib.request
        import json

        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,paperId,citationCount"
        }

        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        request = urllib.request.Request(url)
        if api_key:
            request.add_header("x-api-key", api_key)

        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())

        papers = data.get('data', [])
        if not papers:
            return f"No papers found for query: {query}"

        output = f"Found {len(papers)} papers for '{query}':\n\n"
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'No title')
            authors = ', '.join([a.get('name', '') for a in paper.get('authors', [])[:3]])
            year = paper.get('year', 'Unknown')
            paper_id = paper.get('paperId', 'Unknown')
            citations = paper.get('citationCount', 0)
            abstract = paper.get('abstract', 'No abstract available')

            output += f"{i}. {title}\n"
            output += f"   Authors: {authors} ({year})\n"
            output += f"   Paper ID: {paper_id}\n"
            output += f"   Citations: {citations}\n"
            output += f"   Abstract: {abstract[:300]}...\n\n" if len(abstract) > 300 else f"   Abstract: {abstract}\n\n"

        return output

    except Exception as e:
        return f"Error searching Semantic Scholar: {str(e)}. Note: API key may be required (set SEMANTIC_SCHOLAR_API_KEY env var)."


@mcp.tool()
def list_artifacts() -> str:
    """List available artifacts (figures, data files) that can be included in the paper.

    Returns a list of available artifact files in the artifacts directory.
    """
    paper_path = get_paper_path()
    artifacts_dir = paper_path.parent

    artifacts: list[str] = []

    # Look for common artifact file types
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.svg', '*.csv', '*.json']:
        artifacts.extend([str(p.relative_to(artifacts_dir)) for p in artifacts_dir.rglob(ext)])

    if not artifacts:
        return "No artifacts found in the artifacts directory."

    # Group by directory
    output = f"Found {len(artifacts)} artifacts:\n\n"
    for artifact in sorted(artifacts)[:50]:  # Limit to 50 for readability
        output += f"  - {artifact}\n"

    if len(artifacts) > 50:
        output += f"\n... and {len(artifacts) - 50} more"

    return output


@mcp.tool()
def include_figure(figure_path: str, caption: str, label: str | None = None, width: str = "0.8\\textwidth") -> str:
    """Generate LaTeX figure code to include a figure in the paper.

    Args:
        figure_path: Path to the figure file (relative to paper directory or absolute)
        caption: Caption text for the figure
        label: Optional LaTeX label for referencing (e.g., "fig:results")
        width: Figure width (default: "0.8\\textwidth")

    Returns LaTeX code for a figure environment.
    """
    paper_path = get_paper_path()

    # Normalize path
    if not Path(figure_path).is_absolute():
        # Try relative to paper directory
        full_path = paper_path.parent / figure_path
        if not full_path.exists():
            # Try relative to artifacts root
            full_path = paper_path.parent.parent / figure_path
        figure_path = str(full_path.relative_to(paper_path.parent))

    latex_code = "\\begin{figure}[htbp]\n"
    latex_code += "  \\centering\n"
    latex_code += f"  \\includegraphics[width={width}]{{{figure_path}}}\n"
    latex_code += f"  \\caption{{{caption}}}\n"
    if label:
        latex_code += f"  \\label{{{label}}}\n"
    latex_code += "\\end{figure}"

    return latex_code


@mcp.tool()
def get_paper_template(paper_type: str = "paper") -> str:
    """Get a LaTeX template for a research paper.

    Args:
        paper_type: Type of paper template ("paper", "arxiv", "technical_report", "blog_post")

    Returns a complete LaTeX document template.
    """
    templates = {
        "paper": """\\documentclass[11pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath,amssymb,amsthm}
\\usepackage{graphicx}
\\usepackage{hyperref}
\\usepackage{natbib}

\\title{Your Paper Title}
\\author{Author Name}
\\date{\\today}

\\begin{document}

\\maketitle

\\begin{abstract}
Your abstract here.
\\end{abstract}

\\section{Introduction}
Introduction content.

\\section{Related Work}
Related work content.

\\section{Methodology}
Methodology content.

\\section{Experiments}
Experiments content.

\\section{Results}
Results content.

\\section{Discussion}
Discussion content.

\\section{Conclusion}
Conclusion content.

\\bibliographystyle{plainnat}
\\bibliography{references}

\\end{document}
""",
        "arxiv": """\\documentclass[11pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath,amssymb}
\\usepackage{graphicx}
\\usepackage{hyperref}
\\usepackage{natbib}

\\title{Your Paper Title}
\\author{Author Name\\\\
Institution\\\\
\\texttt{email@example.com}}

\\begin{document}

\\maketitle

\\begin{abstract}
Your abstract here.
\\end{abstract}

\\section{Introduction}
Introduction content.

\\section{Method}
Method content.

\\section{Experiments}
Experiments content.

\\section{Results}
Results content.

\\section{Conclusion}
Conclusion content.

\\bibliographystyle{plainnat}
\\bibliography{references}

\\end{document}
""",
    }

    return templates.get(paper_type.lower(), templates["paper"])


# Alias for backward compatibility
@mcp.tool()
def read_paper() -> str:
    """Read a paper from the paper database (alias for read_tex_file)."""
    return read_tex_file()


@mcp.tool()
def write_paper(paper: str) -> str:
    """Write a paper to the paper database (alias for write_tex_file).

    Args:
        paper: The LaTeX content of the paper to write.
    """
    return write_tex_file(paper)


@mcp.tool()
def cite_source(source: str) -> str:
    """Cite a source for the paper (alias for add_citation).

    Args:
        source: The BibTeX entry to add.
    """
    return add_citation(source)


# Backward compatibility wrapper for stdio transport
class PaperTool:
    """Paper writing tools MCP server (backward compatibility wrapper)."""

    def __init__(self, paper_path: Path | None = None):
        # For backward compatibility, but not used in containerized setup
        pass

    def get_command(self) -> str:
        return sys.executable

    def get_args(self) -> list[str]:
        return ["-m", "agent.tools.paper"]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")