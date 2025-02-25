/*
Package memory provides memory-related functionality for the agent system.
This file contains the schema definitions for memory extraction.
*/
package memory

// MemoryDocument represents a document memory to be stored
type MemoryDocument struct {
	Document string                 `json:"document" jsonschema_description:"The complete text of an important memory to store"`
	Metadata map[string]interface{} `json:"metadata" jsonschema_description:"Additional information about this memory"`
}

// MemoryEntity represents an entity and its relationship to store in the graph database
type MemoryEntity struct {
	Entity       string                 `json:"entity" jsonschema_description:"The name or identifier of the entity"`
	Relationship string                 `json:"relationship" jsonschema_description:"The relationship type, such as RELATES_TO, PART_OF, etc."`
	Target       string                 `json:"target" jsonschema_description:"The target entity that this entity relates to"`
	Metadata     map[string]interface{} `json:"metadata" jsonschema_description:"Additional information about this entity relationship"`
}

// MemoryExtraction represents the complete memory extraction result
type MemoryExtraction struct {
	Documents []MemoryDocument `json:"documents" jsonschema_description:"Document-like memories to store in the vector database"`
	Entities  []MemoryEntity   `json:"entities" jsonschema_description:"Entity relationships to store in the graph database"`
}

// MemoryQueries represents the memory query generation result
type MemoryQueries struct {
	Queries []string `json:"queries" jsonschema_description:"A list of search queries to retrieve relevant memories"`
}
