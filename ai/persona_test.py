"""
Unit tests for the persona module.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from caramba.ai.persona import Persona


# Expected output schema for knowledge_curator persona
EXPECTED_KNOWLEDGE_CURATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "temp_id": {
                        "type": "string",
                        "description": "Descriptive identifier for referencing this entity (e.g., 'concept_kv_cache', 'decision_use_rotary')"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["Concept", "Person", "Decision", "Fact", "Insight"],
                        "description": "Type of entity"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name or title of the entity"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description sufficient for someone unfamiliar with the conversation"
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
                        "description": "temp_id of the source entity"
                    },
                    "target": {
                        "type": "string",
                        "description": "temp_id of the target entity"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["RELATES_TO", "DEPENDS_ON", "LEADS_TO", "CONTRADICTS", "REFINES", "SUPPORTS"],
                        "description": "Type of relationship"
                    },
                    "description": {
                        "type": "string",
                        "description": "Explanation of why this relationship exists"
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
                        "description": "The factual statement"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context sufficient to understand the fact independently"
                    },
                    "source": {
                        "type": "string",
                        "description": "Who stated this, if attributed"
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
        "facts"
    ]
}


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
            self.assertEqual(persona.output_schema, EXPECTED_KNOWLEDGE_CURATOR_SCHEMA)
