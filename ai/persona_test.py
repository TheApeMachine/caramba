"""
Unit tests for the persona module.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from caramba.ai.persona import Persona


class TestPersona(unittest.TestCase):
    """Tests for the persona module."""

    def test_persona_from_yaml(self) -> None:
        """Test that a persona can be loaded from a YAML file."""
        persona = Persona.from_yaml(Path("config/personas/knowledge_curator.yml"))
        self.assertEqual(persona.name, "Knowledge_Curator")
        self.assertEqual(persona.description, "A specialized agent that extracts structured knowledge, entities, and relationships from conversations.")
        self.assertEqual(persona.model, "gemini/gemini-3-pro-preview")
        self.assertEqual(persona.temperature, 0.1)
        self.assertEqual(persona.tools, [])

        if persona.output_schema is not None:
            self.assertEqual(persona.output_schema, {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "temp_id": {
                                    "type": "string",
                                    "description": "Temporary identifier for referencing this entity (can be any string like 'entity1', 'concept_a', etc.)"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Type of entity: Concept, Person, Decision, Fact, or Insight"
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Name or title of the entity"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Detailed description of the entity"
                                }
                            },
                            "required": [
                                "temp_id",
                                "type",
                                "name",
                                "description"
                            ]
                        }
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "Temporary ID of the source entity (must match temp_id from entities list)"
                                },
                                "target": {
                                    "type": "string",
                                    "description": "Temporary ID of the target entity (must match temp_id from entities list)"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Type of relationship: RELATES_TO, DEPENDS_ON, LEADS_TO, or SIMILAR_TO"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of the relationship"
                                }
                            },
                            "required": [
                                "source",
                                "target",
                                "type",
                                "description"
                            ]
                        }
                    },
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The factual content"
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Context in which this fact was mentioned"
                                }
                            },
                            "required": [
                                "content",
                                "context"
                            ]
                        }
                    }
                },
                "required": [
                    "entities",
                    "relationships",
                    "facts",
                ]
            })
