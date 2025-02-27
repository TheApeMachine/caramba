/*
Package memory provides memory-related functionality for the agent system.
This file contains the schema definitions for memory extraction.
*/
package memory

// MemoryDocument represents a document memory to be stored
type MemoryDocument struct {
	Document string            `json:"document" jsonschema:"The complete text of an important memory to store,required"`
	Metadata map[string]string `json:"metadata" jsonschema:"Additional information about this memory"`
}

// MemoryEntity represents an entity and its relationship to store in the graph database
type MemoryEntity struct {
	Entity       string            `json:"entity" jsonschema:"The name or identifier of the entity,required"`
	Relationship string            `json:"relationship" jsonschema:"The relationship type, such as RELATES_TO, PART_OF, etc.,required"`
	Target       string            `json:"target" jsonschema:"The target entity that this entity relates to,required"`
	Metadata     map[string]string `json:"metadata" jsonschema:"Additional information about this entity relationship"`
}

// MemoryExtraction represents the complete memory extraction result
type MemoryExtraction struct {
	Documents []MemoryDocument `json:"documents" jsonschema:"Document-like memories to store in the vector database,required"`
	Entities  []MemoryEntity   `json:"entities" jsonschema:"Entity relationships to store in the graph database,required"`
}

// MemoryQueries represents the memory query generation result
type MemoryQueries struct {
	Queries []string `json:"queries" jsonschema:"A list of search queries to retrieve relevant memories,required"`
}

// EnhancedMemoryEntryExtensions adds additional fields to the EnhancedMemoryEntry struct
// These additional fields are used for memory prioritization and optimization
type EnhancedMemoryEntryExtensions struct {
	// ImportanceScore is the algorithmically determined importance of this memory
	ImportanceScore float32
	// RelevanceCache caches relevance scores to common queries
	RelevanceCache *RelevanceCache
	// UsageStats tracks detailed usage statistics for the memory
	UsageStats *MemoryUsageStats
}
