package memory

import (
	"context"
	"errors"
	"fmt"

	"github.com/theapemachine/caramba/pkg/agent/core"
)

/*
Relationship represents a relationship between two memory entries in a graph.
It defines connections between memory entries, allowing for knowledge graph construction.
*/
type Relationship struct {
	/* FromID is the ID of the source memory */
	FromID string
	/* ToID is the ID of the target memory */
	ToID string
	/* Type identifies the kind of relationship */
	Type string
	/* Metadata contains additional information about the relationship */
	Metadata map[string]string
}

/*
ToCoreRelationship converts to core.Relationship from local Relationship.
This method bridges between the memory package's types and the core package's relationship types.

Returns:
  - The converted core.Relationship
*/
func (r *Relationship) ToCoreRelationship() core.Relationship {
	return core.Relationship{
		FromID:   r.FromID,
		ToID:     r.ToID,
		Type:     r.Type,
		Metadata: r.Metadata,
	}
}

/*
FromCoreRelationship converts from core.Relationship to local Relationship.
This function bridges between the core package's relationship types and the memory package's types.

Parameters:
  - rel: The core package's relationship to convert

Returns:
  - The converted local Relationship
*/
func FromCoreRelationship(rel core.Relationship) Relationship {
	return Relationship{
		FromID:   rel.FromID,
		ToID:     rel.ToID,
		Type:     rel.Type,
		Metadata: rel.Metadata,
	}
}

/*
CreateRelationship creates a relationship between two memories.
It establishes a typed connection between memory nodes in the graph store.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - relationship: The relationship to create

Returns:
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) CreateRelationship(ctx context.Context, relationship Relationship) error {
	if !um.options.EnableGraphStore || um.graphStore == nil {
		return errors.New("graph store not enabled or not available")
	}

	// Create relationship in the graph store
	return um.graphStore.CreateRelationship(
		ctx,
		relationship.FromID,
		relationship.ToID,
		relationship.Type,
		relationship.Metadata,
	)
}

/*
GetRelatedMemories retrieves memories related to a specific memory.
It finds connected memories in the graph store based on relationship types.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - memoryID: The ID of the memory to find relations for
  - relationshipType: The type of relationship to follow (empty for any)
  - maxDepth: The maximum traversal depth in the graph

Returns:
  - A slice of EnhancedMemoryEntry objects related to the specified memory
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) GetRelatedMemories(ctx context.Context, memoryID string, relationshipType string, maxDepth int) ([]EnhancedMemoryEntry, error) {
	if !um.options.EnableGraphStore || um.graphStore == nil {
		return nil, errors.New("graph store not enabled or not available")
	}

	// Default max depth
	if maxDepth <= 0 {
		maxDepth = 1
	}

	// Build Cypher query to find related memories
	var query string
	params := map[string]interface{}{
		"memoryID": memoryID,
	}

	if relationshipType != "" {
		query = fmt.Sprintf(
			`MATCH (m:Memory {id: $memoryID})-[r:%s*1..%d]-(related:Memory)
			 RETURN related as memory`,
			relationshipType, maxDepth,
		)
	} else {
		query = fmt.Sprintf(
			`MATCH (m:Memory {id: $memoryID})-[r*1..%d]-(related:Memory)
			 RETURN related as memory`,
			maxDepth,
		)
	}

	// Execute query and convert results
	return um.RetrieveMemoriesByGraph(ctx, query, params)
}
