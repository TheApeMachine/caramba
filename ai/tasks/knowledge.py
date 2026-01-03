"""Knowledge extraction task.

This task is used to extract knowledge from the conversation history.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field

from falkordb import FalkorDB, Graph
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from caramba.ai.agent import Agent
from caramba.ai.persona import Persona
from caramba.ai.tasks.task import Task
from caramba.console import logger


class Entity(BaseModel):
    """An entity extracted from the conversation."""
    temp_id: str = Field(description="Temporary identifier for referencing this entity (can be any string like 'entity1', 'concept_a', etc.)")
    type: str = Field(description="Type of entity: Concept, Person, Decision, Fact, or Insight")
    name: str = Field(description="Name or title of the entity")
    description: str = Field(description="Detailed description of the entity")


class Relationship(BaseModel):
    """A relationship between two entities."""
    source: str = Field(description="Temporary ID of the source entity (must match temp_id from entities list)")
    target: str = Field(description="Temporary ID of the target entity (must match temp_id from entities list)")
    type: str = Field(description="Type of relationship: RELATES_TO, DEPENDS_ON, LEADS_TO, or SIMILAR_TO")
    description: str = Field(description="Description of the relationship")


class Fact(BaseModel):
    """A fact extracted from the conversation."""
    content: str = Field(description="The factual content")
    context: str = Field(description="Context in which this fact was mentioned")


class KnowledgeExtractionOutput(BaseModel):
    """Structured output schema for knowledge extraction."""
    entities: list[Entity] = Field(default_factory=list, description="List of extracted entities")
    relationships: list[Relationship] = Field(default_factory=list, description="List of relationships between entities")
    facts: list[Fact] = Field(default_factory=list, description="List of extracted facts")


class KnowledgeExtractionTask(Task):
    """Knowledge extraction task that extracts structured knowledge from conversation history and stores it in FalkorDB."""
    def __init__(
        self,
        conversation_history: list[dict[str, Any]],
        persona_path: Path | None = None,
        prompt_path: Path | None = None,
        graph_name: str = "caramba_knowledge_base",
    ):
        super().__init__("knowledge_extraction")
        self.conversation_history = conversation_history
        self.persona_path = persona_path or Path("config/personas/knowledge_curator.yml")
        self.prompt_path = prompt_path or Path("config/prompts/knowledge.yml")
        self.graph_name = graph_name

    def _render_history(self) -> str:
        """Render conversation history as markdown."""
        lines: list[str] = []
        for msg in self.conversation_history:
            mtype = str(msg.get("type", ""))
            author = str(msg.get("author", ""))
            content = msg.get("content", "")
            if mtype == "user":
                lines.append(f"- **{author}**: {content}")
            elif mtype == "assistant":
                lines.append(f"- **{author}**: {content}")
            elif mtype == "tool_call":
                payload = content if isinstance(content, dict) else {"call": content}
                lines.append(
                    f"- **{author} tool call**\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
                )
            elif mtype == "tool_result":
                payload = content if isinstance(content, dict) else {"result": content}
                lines.append(
                    f"- **tool result**\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
                )
            else:
                lines.append(f"- **{author}** ({mtype}): {content}")
        return "\n".join(lines).strip()

    def _connect_falkordb(self) -> Graph:
        """Connect to FalkorDB and return the knowledge base graph."""
        uri = os.getenv("FALKORDB_URI") or os.getenv("FALKOR_URI")
        password = os.getenv("FALKORDB_PASSWORD") or None

        if uri:
            client = FalkorDB(url=uri, password=password)
        else:
            host = os.getenv("FALKORDB_HOST") or "localhost"
            port = int(os.getenv("FALKORDB_PORT") or 6379)
            client = FalkorDB(host=host, port=port, password=password)

        return Graph(client, self.graph_name)

    def _load_prompt(self) -> str:
        """Load the extraction prompt from YAML config."""
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        with open(self.prompt_path, "r") as f:
            data = yaml.safe_load(f) or {}
            knowledge_config = data.get("knowledge_extraction", {})
            prompt_template = knowledge_config.get("extraction_prompt", "")
            if not prompt_template:
                raise ValueError(f"No extraction_prompt found in {self.prompt_path}")
            return prompt_template

    async def _extract_knowledge(self) -> KnowledgeExtractionOutput | None:
        """Extract useful knowledge from conversation history using an AI agent with output_schema."""
        if not self.conversation_history:
            return None

        # Load persona from YAML
        if not self.persona_path.exists():
            raise FileNotFoundError(f"Persona file not found: {self.persona_path}")

        persona = Persona.from_yaml(self.persona_path)

        # Create standard Agent wrapper
        # The output_schema is now loaded from the persona YAML and passed to the underlying LlmAgent
        agent = Agent(
            persona=persona,
            app_name="knowledge_extraction",
            user_id="system",
        )

        # Render full history for extraction
        full_transcript = self._render_history()

        # Load prompt template and format it
        prompt_template = self._load_prompt()
        extraction_prompt = prompt_template.format(conversation_history=full_transcript)

        try:
            # Run the agent asynchronously
            # agent.run_async returns the raw text response
            final_response = await agent.run_async(extraction_prompt)

            if not final_response:
                return None

            # Parse the JSON response (output_schema ensures it's valid JSON)
            parsed = json.loads(final_response)
            return KnowledgeExtractionOutput.model_validate(parsed)

        except Exception as e:
            logger.warning(f"Knowledge extraction failed: {e}")
            return None

    def _find_existing_entity(self, graph: Graph, entity_type: str, name: str, description: str) -> str | None:
        """Find an existing entity in the graph by name and description similarity."""
        # First try exact name match
        query = f"MATCH (n:{entity_type} {{name: $name}}) RETURN n.id as id LIMIT 1"
        result = graph.query(query, {"name": name})
        if result.result_set:
            return result.result_set[0][0]

        # If no exact match, try fuzzy matching on name (case-insensitive)
        query = f"MATCH (n:{entity_type}) WHERE toLower(n.name) = toLower($name) RETURN n.id as id LIMIT 1"
        result = graph.query(query, {"name": name})
        if result.result_set:
            return result.result_set[0][0]

        return None

    def _resolve_entity_ids(
        self, graph: Graph, entities: list[Entity]
    ) -> dict[str, str]:
        """Resolve temporary entity IDs to real UUIDs, deduplicating against existing entities."""
        temp_to_real: dict[str, str] = {}

        for entity in entities:
            # Check if entity already exists in graph
            existing_id = self._find_existing_entity(graph, entity.type, entity.name, entity.description)

            if existing_id:
                # Use existing ID
                temp_to_real[entity.temp_id] = existing_id
            else:
                # Create new UUID
                real_id = str(uuid4())
                temp_to_real[entity.temp_id] = real_id

        return temp_to_real

    def _store_knowledge_in_falkordb(self, knowledge: KnowledgeExtractionOutput) -> None:
        """Store extracted knowledge into FalkorDB graph."""
        if not knowledge:
            return

        try:
            graph = self._connect_falkordb()

            # Resolve temporary IDs to real UUIDs, deduplicating against existing entities
            temp_to_real_id = self._resolve_entity_ids(graph, knowledge.entities)

            # Store entities as nodes (using real IDs)
            for entity in knowledge.entities:
                real_id = temp_to_real_id[entity.temp_id]
                props = {
                    "id": real_id,
                    "name": entity.name,
                    "description": entity.description,
                    "extracted_at": time.time(),
                }

                # Use MERGE to upsert nodes (will update if exists, create if not)
                query = f"MERGE (n:{entity.type} {{id: $id}}) SET n += $props"
                graph.query(query, {"id": real_id, "props": props})

            # Store relationships (using real IDs)
            for rel in knowledge.relationships:
                source_real_id = temp_to_real_id.get(rel.source)
                target_real_id = temp_to_real_id.get(rel.target)

                if not source_real_id or not target_real_id:
                    logger.warning(f"Skipping relationship with invalid temp IDs: {rel.source} -> {rel.target}")
                    continue

                query = (
                    "MATCH (a {id: $source_id}), (b {id: $target_id}) "
                    f"MERGE (a)-[r:{rel.type}]->(b) "
                    "SET r.description = $description, r.extracted_at = $extracted_at"
                )
                graph.query(
                    query,
                    {
                        "source_id": source_real_id,
                        "target_id": target_real_id,
                        "description": rel.description,
                        "extracted_at": time.time(),
                    },
                )

            # Store facts as Fact nodes (check for duplicates by content)
            for fact in knowledge.facts:
                # Check if fact already exists
                query = "MATCH (f:Fact {content: $content}) RETURN f.id as id LIMIT 1"
                result = graph.query(query, {"content": fact.content})
                if result.result_set:
                    # Fact already exists, skip
                    continue

                fact_id = str(uuid4())
                query = "CREATE (f:Fact {id: $id, content: $content, context: $context, extracted_at: $extracted_at})"
                graph.query(
                    query,
                    {
                        "id": fact_id,
                        "content": fact.content,
                        "context": fact.context,
                        "extracted_at": time.time(),
                    },
                )

            total_items = len(knowledge.entities) + len(knowledge.relationships) + len(knowledge.facts)
            logger.info(
                f"Stored {total_items} knowledge items ({len(knowledge.entities)} entities, {len(knowledge.relationships)} relationships, {len(knowledge.facts)} facts) into {self.graph_name}"
            )

        except Exception as e:
            logger.warning(f"Failed to store knowledge in FalkorDB: {e}")

    async def run_async(self) -> dict[str, Any]:
        """Run the knowledge extraction task asynchronously."""
        logger.info("Extracting knowledge from conversation history...")

        # Extract knowledge using output_schema
        knowledge = await self._extract_knowledge()

        if knowledge:
            logger.info(f"Storing extracted knowledge in FalkorDB graph '{self.graph_name}'...")
            self._store_knowledge_in_falkordb(knowledge)
            return {"success": True, "knowledge_extracted": True}
        else:
            logger.info("No knowledge extracted (empty history or extraction failed)")
            return {"success": True, "knowledge_extracted": False}