"""Graph memory management for the caramba system.

Provides tools for ingesting codebases into a graph database (FalkorDB)
and performing AST-based analysis.
"""
from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from console import logger

@dataclass(frozen=True)
class Symbol:
    """A code symbol (class, function, etc.) extracted from AST."""
    name: str
    type: str  # 'class', 'function', 'async_function', 'module'
    line: int
    content: str
    docstring: str | None = None


class CodeParser:
    """Parser for Python source code using AST."""

    def parse_file(self, file_path: Path) -> tuple[list[Symbol], list[tuple[str, str, str]]]:
        """Parse a Python file and return its symbols and relationships.

        Returns:
            A tuple of (symbols, relationships).
            Relationships are (source, target, type) tuples.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return [], []

        symbols: list[Symbol] = []
        relationships: list[tuple[str, str, str]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
                if isinstance(node, ast.AsyncFunctionDef):
                    symbol_type = "async_function"

                # Extract content (roughly)
                lines = content.splitlines()
                # AST line numbers are 1-indexed
                start_line = node.lineno - 1
                end_line = node.end_lineno or (start_line + 1)
                node_content = "\n".join(lines[start_line:end_line])

                symbols.append(Symbol(
                    name=node.name,
                    type=symbol_type,
                    line=node.lineno,
                    content=node_content,
                    docstring=ast.get_docstring(node)
                ))

                # If this is a class, find its methods and link them
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            relationships.append((item.name, node.name, "method_of"))

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    relationships.append((alias.name, "module", "imports"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or "unknown"
                for alias in node.names:
                    relationships.append((alias.name, module, "imports_from"))

        return symbols, relationships


class DirectGraphWriter:
    """Direct writer for FalkorDB graph memory."""

    def __init__(self, host: str = "localhost", port: int = 6379, graph_name: str = "main"):
        self._host = host
        self._port = port
        self._graph_name = graph_name
        self._client: Any = None
        self._graph: Any = None
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if the graph database is available and initialized."""
        if not self._initialized:
            self._connect()
        return self._initialized

    def _connect(self) -> None:
        """Connect to FalkorDB."""
        try:
            from falkordb import FalkorDB
            self._client = FalkorDB(host=self._host, port=self._port)
            self._graph = self._client.select_graph(self._graph_name)
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB: {e}")
            self._initialized = False

    def add_module(self, name: str, path: str) -> None:
        """Add a module node to the graph."""
        if not self.is_available: return
        query = "MERGE (m:Module {name: $name}) SET m.path = $path"
        self._graph.query(query, {"name": name, "path": path})

    def add_symbol(self, symbol: Symbol) -> None:
        """Add a symbol node to the graph."""
        if not self.is_available: return
        query = """
        MERGE (s:Symbol {name: $name})
        SET s.type = $type, s.line = $line, s.content = $content, s.docstring = $docstring
        """
        self._graph.query(query, {
            "name": symbol.name,
            "type": symbol.type,
            "line": symbol.line,
            "content": symbol.content,
            "docstring": symbol.docstring or ""
        })

    def link_symbol_to_module(self, symbol_name: str, module_name: str) -> None:
        """Create a relationship between a symbol and its module."""
        if not self.is_available: return
        query = """
        MATCH (s:Symbol {name: $s_name}), (m:Module {name: $m_name})
        MERGE (s)-[:DEFINED_IN]->(m)
        """
        self._graph.query(query, {"s_name": symbol_name, "m_name": module_name})

    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        *,
        source_type: str | None = None,
        target_type: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add a typed relationship between two nodes with full graph semantics.

        Args:
            source: Name of the source node
            target: Name of the target node
            rel_type: Relationship type (e.g., 'CALLS', 'IMPORTS', 'INHERITS')
            source_type: Optional node label for source (e.g., 'Symbol', 'Module')
            target_type: Optional node label for target
            properties: Optional properties to attach to the relationship
        """
        if not self.is_available:
            return

        # Build node match patterns with optional type constraints
        if source_type:
            src_pattern = f"(src:{source_type} {{name: $source}})"
        else:
            src_pattern = "(src {name: $source})"

        if target_type:
            tgt_pattern = f"(tgt:{target_type} {{name: $target}})"
        else:
            tgt_pattern = "(tgt {name: $target})"

        # Build relationship with optional properties
        if properties:
            # Construct property string for relationship
            prop_assignments = ", ".join(
                f"r.{k} = ${k}" for k in properties.keys()
            )
            query = f"""
            MERGE {src_pattern}
            MERGE {tgt_pattern}
            MERGE (src)-[r:{rel_type}]->(tgt)
            SET {prop_assignments}
            RETURN r
            """
            params = {"source": source, "target": target, **properties}
        else:
            query = f"""
            MERGE {src_pattern}
            MERGE {tgt_pattern}
            MERGE (src)-[r:{rel_type}]->(tgt)
            RETURN r
            """
            params = {"source": source, "target": target}

        try:
            self._graph.query(query, params)
        except Exception as e:
            logger.warning(f"Failed to add relationship {source} -[{rel_type}]-> {target}: {e}")

    def add_typed_node(
        self,
        name: str,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add a node with a specific label and properties.

        Args:
            name: Node identifier
            label: Node label (e.g., 'Class', 'Function', 'Variable')
            properties: Additional properties to set on the node
        """
        if not self.is_available:
            return

        if properties:
            prop_assignments = ", ".join(
                f"n.{k} = ${k}" for k in properties.keys()
            )
            query = f"""
            MERGE (n:{label} {{name: $name}})
            SET {prop_assignments}
            RETURN n
            """
            params = {"name": name, **properties}
        else:
            query = f"""
            MERGE (n:{label} {{name: $name}})
            RETURN n
            """
            params = {"name": name}

        try:
            self._graph.query(query, params)
        except Exception as e:
            logger.warning(f"Failed to add node {label}:{name}: {e}")

    def find_paths(
        self,
        source: str,
        target: str,
        *,
        max_depth: int = 5,
        rel_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find all paths between two nodes.

        Args:
            source: Starting node name
            target: Ending node name
            max_depth: Maximum path length
            rel_types: Optional list of relationship types to traverse

        Returns:
            List of paths, where each path is a list of node dicts
        """
        if not self.is_available:
            return []

        if rel_types:
            rel_filter = "|".join(rel_types)
            rel_pattern = f"[*1..{max_depth} {{{rel_filter}}}]"
        else:
            rel_pattern = f"[*1..{max_depth}]"

        query = f"""
        MATCH path = (src {{name: $source}})-{rel_pattern}-(tgt {{name: $target}})
        RETURN [node IN nodes(path) | properties(node)] AS path_nodes
        LIMIT 100
        """

        try:
            result = self._graph.query(query, {"source": source, "target": target})
            return [row[0] for row in result.result_set]
        except Exception as e:
            logger.warning(f"Failed to find paths from {source} to {target}: {e}")
            return []

    def get_neighbors(
        self,
        node_name: str,
        *,
        direction: str = "both",
        rel_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes.

        Args:
            node_name: Node to find neighbors of
            direction: 'in', 'out', or 'both'
            rel_types: Optional relationship types to filter
            limit: Maximum neighbors to return

        Returns:
            List of neighbor node properties
        """
        if not self.is_available:
            return []

        if rel_types:
            rel_filter = ":" + "|".join(rel_types)
        else:
            rel_filter = ""

        if direction == "out":
            pattern = f"(n {{name: $name}})-[r{rel_filter}]->(neighbor)"
        elif direction == "in":
            pattern = f"(n {{name: $name}})<-[r{rel_filter}]-(neighbor)"
        else:
            pattern = f"(n {{name: $name}})-[r{rel_filter}]-(neighbor)"

        query = f"""
        MATCH {pattern}
        RETURN properties(neighbor) AS neighbor, type(r) AS rel_type
        LIMIT {limit}
        """

        try:
            result = self._graph.query(query, {"name": node_name})
            return [
                {**row[0], "_rel_type": row[1]}
                for row in result.result_set
            ]
        except Exception as e:
            logger.warning(f"Failed to get neighbors of {node_name}: {e}")
            return []

    def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Execute a raw Cypher query.

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            Query result set
        """
        if not self.is_available:
            return []

        try:
            result = self._graph.query(query, params or {})
            return result.result_set
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []

    def get_subgraph(
        self,
        center_node: str,
        *,
        max_depth: int = 2,
        max_nodes: int = 100,
    ) -> dict[str, Any]:
        """Extract a subgraph centered on a node.

        Args:
            center_node: Node to center the subgraph on
            max_depth: Maximum distance from center
            max_nodes: Maximum nodes to include

        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        if not self.is_available:
            return {"nodes": [], "edges": []}

        query = f"""
        MATCH path = (center {{name: $name}})-[*0..{max_depth}]-(connected)
        WITH DISTINCT connected
        LIMIT {max_nodes}
        MATCH (connected)-[r]-(other)
        WHERE other IN [connected]
        RETURN
            COLLECT(DISTINCT properties(connected)) AS nodes,
            COLLECT(DISTINCT {{
                source: startNode(r).name,
                target: endNode(r).name,
                type: type(r)
            }}) AS edges
        """

        try:
            result = self._graph.query(query, {"name": center_node})
            if result.result_set:
                row = result.result_set[0]
                return {"nodes": row[0] or [], "edges": row[1] or []}
            return {"nodes": [], "edges": []}
        except Exception as e:
            logger.warning(f"Failed to get subgraph for {center_node}: {e}")
            return {"nodes": [], "edges": []}

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        if not self.is_available: return {}
        try:
            # Simple stats check
            nodes = self._graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
            rels = self._graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
            return {"nodes": nodes, "relationships": rels}
        except Exception:
            return {}

    def close(self) -> None:
        """Close the database connection."""
        if self._client:
            # falkordb-py doesn't have an explicit close for the client usually,
            # but we can at least null out the references.
            self._client = None
            self._graph = None
            self._initialized = False


class CodebaseIngestor:
    """Orchestrator for codebase ingestion into graph memory."""

    def __init__(
        self,
        source_dir: Path,
        falkordb_host: str | None = None,
        falkordb_port: int | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ):
        self.source_dir = source_dir.resolve()
        self.writer = DirectGraphWriter(
            host=falkordb_host or "localhost",
            port=falkordb_port or 6379
        )
        self.parser = CodeParser()
        self.include_patterns = include_patterns or ["**/*.py"]
        self.exclude_patterns = exclude_patterns or []

        # Add default exclusions
        self.exclude_patterns.extend([
            "**/.git/**", "**/.venv/**", "**/__pycache__/**",
            "**/.pytest_cache/**", "**/node_modules/**"
        ])

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded based on patterns."""
        rel_path = str(file_path.relative_to(self.source_dir))
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path, pattern.lstrip("*/")):
                return True
            # Check path parts for directory matches
            for part in file_path.relative_to(self.source_dir).parts:
                if fnmatch.fnmatch(part, pattern.replace("**/", "").replace("/**", "")):
                    return True
        return False

    def _collect_files(self) -> list[Path]:
        """Collect all files matching include patterns that are not excluded."""
        files = []
        for pattern in self.include_patterns:
            for file_path in self.source_dir.glob(pattern):
                if file_path.is_file() and not self._should_exclude(file_path):
                    files.append(file_path)
        return files

    def ingest(self) -> dict[str, Any]:
        """Perform the ingestion.

        Returns a dict with ingestion statistics.
        """
        if not self.writer.is_available:
            raise RuntimeError("Graph database not available")

        files = self._collect_files()
        logger.info(f"Found {len(files)} files to process")

        total_symbols = 0
        total_relationships = 0
        modules_added = set()

        for i, file_path in enumerate(files):
            rel_path = file_path.relative_to(self.source_dir)
            module_name = str(rel_path).replace("/", ".").replace(".py", "")

            if module_name not in modules_added:
                self.writer.add_module(module_name, str(rel_path))
                modules_added.add(module_name)

            symbols, relationships = self.parser.parse_file(file_path)

            for symbol in symbols:
                self.writer.add_symbol(symbol)
                total_symbols += 1
                self.writer.link_symbol_to_module(symbol.name, module_name)

            for source, target, rel_type in relationships:
                self.writer.add_relationship(source, target, rel_type)
                total_relationships += 1

            if (i + 1) % 10 == 0 or i == len(files) - 1:
                logger.step(i + 1, len(files), f"Processed {rel_path}")

        stats = self.writer.get_stats()
        self.writer.close()

        return {
            "symbols": total_symbols,
            "relationships": total_relationships,
            "nodes": stats.get("nodes", 0),
            "edges": stats.get("relationships", 0)
        }
