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
from google.genai import types

from caramba.ai.agent import Agent
from caramba.ai.persona import Persona
from caramba.ai.tasks import Task
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
    ):
        super().__init__("knowledge_extraction")
        self.agent = Agent(
            persona=Persona.from_yaml(Path("config/personas/knowledge_curator.yml")),
            app_name="knowledge_extraction",
            user_id="system",
        )
        self.graph = Graph(
            FalkorDB(
                url=os.getenv("FALKORDB_URI") or os.getenv("FALKOR_URI")
            ), "caramba_knowledge_base"
        )

    def run(self, history: list[types.Content]) -> dict[str, Any]:
        """Run the task synchronously (wrapper around `run_async()`)."""
        return super().run(history)

    async def extract_knowledge(self) -> KnowledgeExtractionOutput | None:
        """Extract useful knowledge from conversation history using an AI agent with output_schema."""
        if not self.history:
            logger.warning("No conversation history to extract knowledge from.")
            return

        knowledge = await self.agent.run_async(
            types.Content(
                role="user",
                parts=[part for item in self.history if item.parts for part in item.parts]
            )
        )

        try:
            return KnowledgeExtractionOutput.model_validate(knowledge)

        except Exception as e:
            logger.warning(f"Knowledge extraction failed: {e}")
            return None

    def find_existing_entity(self, graph: Graph, entity_type: str, name: str) -> str | None:
        """Find an existing entity in the graph by name."""
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

    def resolve_entity_ids(
        self, graph: Graph, entities: list[Entity]
    ) -> dict[str, str]:
        """Resolve temporary entity IDs to real UUIDs, deduplicating against existing entities."""
        temp_to_real: dict[str, str] = {}

        for entity in entities:
            # Check if entity already exists in graph
            existing_id = self.find_existing_entity(graph, entity.type, entity.name)

            if existing_id:
                # Use existing ID
                temp_to_real[entity.temp_id] = existing_id
            else:
                # Create new UUID
                real_id = str(uuid4())
                temp_to_real[entity.temp_id] = real_id

        return temp_to_real

    def store_knowledge_in_falkordb(self, knowledge: KnowledgeExtractionOutput) -> None:
        """Store extracted knowledge into FalkorDB graph."""
        if not knowledge:
            return

        try:
            # Resolve temporary IDs to real UUIDs, deduplicating against existing entities
            temp_to_real_id = self.resolve_entity_ids(self.graph, knowledge.entities)

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
                self.graph.query(query, {"id": real_id, "props": props})

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
                self.graph.query(
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
                result = self.graph.query(query, {"content": fact.content})
                if result.result_set:
                    # Fact already exists, skip
                    continue

                fact_id = str(uuid4())
                query = "CREATE (f:Fact {id: $id, content: $content, context: $context, extracted_at: $extracted_at})"
                self.graph.query(
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
                f"Stored {total_items} knowledge items ({len(knowledge.entities)} entities, {len(knowledge.relationships)} relationships, {len(knowledge.facts)} facts) into {self.graph.name}"
            )

        except Exception as e:
            logger.warning(f"Failed to store knowledge in FalkorDB: {e}")

    async def run_async(self) -> dict[str, Any]:
        """Run the knowledge extraction task asynchronously."""
        logger.info("Extracting knowledge from conversation history...")

        # Extract knowledge using output_schema
        knowledge = await self.extract_knowledge()

        if knowledge:
            logger.info(f"Storing extracted knowledge in FalkorDB graph '{self.graph.name}'...")
            self.store_knowledge_in_falkordb(knowledge)
            return {"success": True, "knowledge_extracted": True}
        else:
            logger.info("No knowledge extracted (empty history or extraction failed)")
            return {"success": True, "knowledge_extracted": False}