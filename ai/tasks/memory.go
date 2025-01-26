package tasks

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
)

type Memory struct {
	neo4j  *tools.Neo4jStore
	qdrant *tools.QdrantStore
}

type MemoryDocument struct {
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata"`
}

func NewMemory() *Memory {
	return &Memory{
		neo4j:  tools.NewNeo4jStore(),
		qdrant: tools.NewQdrantStore("memories", 1536), // Using 1536 dimensions for OpenAI embeddings
	}
}

func (m *Memory) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]any) {
	// Initialize databases
	if err := m.initializeDatabases(); err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				fmt.Sprintf("Failed to initialize databases: %v", err),
			),
		)
		return
	}

	// Parse documents from the args
	documentsRaw, ok := args["documents"]
	if !ok {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				"No documents provided to store in memory",
			),
		)
		return
	}

	// Convert raw documents to proper format
	var documents []MemoryDocument
	documentsJSON, err := json.Marshal(documentsRaw)
	if err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				fmt.Sprintf("Failed to parse documents: %v", err),
			),
		)
		return
	}

	if err := json.Unmarshal(documentsJSON, &documents); err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				fmt.Sprintf("Failed to parse documents: %v", err),
			),
		)
		return
	}

	var results []string

	// Process each document
	for _, doc := range documents {
		// Store in Qdrant for semantic search
		qdrantArgs := map[string]any{
			"documents": []string{doc.Text},
			"metadata":  doc.Metadata,
		}
		qdrantResult := m.qdrant.Use(ctx.Identity.Ctx, qdrantArgs)
		results = append(results, fmt.Sprintf("Vector store: %s", qdrantResult))

		// Store in Neo4j for structured relationships
		cypher := `
			CREATE (m:Memory {
				content: $content,
				timestamp: datetime()
			})
		`

		// Add all metadata as properties
		params := map[string]any{
			"content": doc.Text,
		}
		for k, v := range doc.Metadata {
			params[k] = v
		}

		// If there's a type in metadata, create a relationship to a Type node
		if docType, ok := doc.Metadata["type"].(string); ok {
			cypher += `
				MERGE (t:Type {name: $type})
				CREATE (m)-[:IS_TYPE]->(t)
			`
			params["type"] = docType
		}

		neo4jArgs := map[string]any{
			"cypher": cypher,
			"params": params,
		}
		neo4jResult := m.neo4j.Use(ctx.Identity.Ctx, neo4jArgs)
		results = append(results, fmt.Sprintf("Graph store: %s", neo4jResult))
	}

	// Report results back to the context
	ctx.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			fmt.Sprintf("Memory storage results:\n%s",
				strings.Join(results, "\n"),
			),
		),
	)
}

func (m *Memory) initializeDatabases() error {
	if err := m.neo4j.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize Neo4j: %w", err)
	}
	if err := m.qdrant.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize Qdrant: %w", err)
	}
	return nil
}
