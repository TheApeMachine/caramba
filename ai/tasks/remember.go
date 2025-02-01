package tasks

import (
	"encoding/json"
	"fmt"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/tools"
)

type Remember struct {
	neo4j  *tools.Neo4jStore
	qdrant *tools.QdrantStore
}

func NewRemember() *Remember {
	return &Remember{
		neo4j:  tools.NewNeo4jStore(),
		qdrant: tools.NewQdrantStore("memories", 1536), // Using 1536 dimensions for OpenAI embeddings
	}
}

type Document struct {
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
}

func (r *Remember) Execute(ctx *drknow.Context, args map[string]any) string {
	// Initialize databases
	if err := r.initializeDatabases(); err != nil {
		ctx.AddIteration(fmt.Sprintf("Database initialization failed: %v", err))
		return ""
	}

	// Verify database clients are properly initialized
	if r.neo4j == nil || r.qdrant == nil {
		ctx.AddIteration("Database clients not properly initialized")
		return ""
	}

	// Process raw string if provided
	if rawStr, ok := args["raw"].(string); ok {
		if err := r.storeRawMemory(ctx, rawStr); err != nil {
			ctx.AddIteration(fmt.Sprintf("Failed to store raw memory: %v", err))
			return ""
		}
	}

	// Process documents if provided
	if docsStr, ok := args["documents"].(string); ok {
		var docs []Document
		if err := json.Unmarshal([]byte(docsStr), &docs); err != nil {
			ctx.AddIteration(fmt.Sprintf("Failed to parse documents: %v", err))
			return ""
		}
		if err := r.storeDocuments(ctx, docs); err != nil {
			ctx.AddIteration(fmt.Sprintf("Failed to store documents: %v", err))
			return ""
		}
	}

	// Process cypher query if provided
	if cypher, ok := args["cypher"].(string); ok {
		if err := r.storeCypherQuery(ctx, cypher); err != nil {
			ctx.AddIteration(fmt.Sprintf("Failed to execute cypher query: %v", err))
			return ""
		}
	}

	ctx.AddIteration("Memory storage successful")

	return ""
}

func (r *Remember) initializeDatabases() error {
	if err := r.neo4j.Initialize(); err != nil {
		return fmt.Errorf("neo4j initialization: %w", err)
	}
	if err := r.qdrant.Initialize(); err != nil {
		return fmt.Errorf("qdrant initialization: %w", err)
	}
	return nil
}

func (r *Remember) storeRawMemory(ctx *drknow.Context, raw string) error {
	// Store in Neo4j as a basic memory node
	cypherArgs := map[string]any{
		"cypher": `
			CREATE (m:Memory {
				content: $content,
				timestamp: datetime()
			})
		`,
		"params": map[string]interface{}{
			"content": raw,
		},
	}
	r.neo4j.Use(ctx.Identity.Ctx, cypherArgs)

	// Store in Qdrant for semantic search
	qdrantArgs := map[string]any{
		"documents": []Document{{
			Content: raw,
			Metadata: map[string]interface{}{
				"type":      "raw",
				"timestamp": "now",
			},
		}},
	}
	r.qdrant.Use(ctx.Identity.Ctx, qdrantArgs)

	return nil
}

func (r *Remember) storeDocuments(ctx *drknow.Context, docs []Document) error {
	// Store documents in Neo4j
	for _, doc := range docs {
		cypherArgs := map[string]any{
			"cypher": `
				CREATE (m:Memory {
					content: $content,
					metadata: $metadata,
					timestamp: datetime()
				})
			`,
			"params": map[string]interface{}{
				"content":  doc.Content,
				"metadata": doc.Metadata,
			},
		}
		r.neo4j.Use(ctx.Identity.Ctx, cypherArgs)
	}

	// Store documents in Qdrant
	qdrantArgs := map[string]any{
		"documents": docs,
	}
	r.qdrant.Use(ctx.Identity.Ctx, qdrantArgs)

	return nil
}

func (r *Remember) storeCypherQuery(ctx *drknow.Context, cypher string) error {
	cypherArgs := map[string]any{
		"cypher": cypher,
	}
	r.neo4j.Use(ctx.Identity.Ctx, cypherArgs)
	return nil
}
