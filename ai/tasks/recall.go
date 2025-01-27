package tasks

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
)

type Recall struct {
	neo4j  *tools.Neo4jQuery
	qdrant *tools.QdrantQuery
}

func NewRecall() *Recall {
	return &Recall{
		neo4j:  tools.NewNeo4jQuery(),
		qdrant: tools.NewQdrantQuery("memories", 1536), // Assuming 1536 dimensions for OpenAI embeddings
	}
}

func (r *Recall) Execute(ctx *drknow.Context, args map[string]any) Bridge {
	// Initialize databases
	if err := r.initializeDatabases(); err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				fmt.Sprintf("Database initialization failed: %v", err),
			),
		)
	}

	// Process input and prepare queries
	queries, err := r.prepareQueries(args)
	if err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				fmt.Sprintf("Query preparation failed: %v", err),
			),
		)
	}

	// Execute searches and collect results
	memories, err := r.executeSearches(ctx, queries)
	if err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				fmt.Sprintf("Search execution failed: %v", err),
			),
		)
	}

	// Marshal results
	result, err := json.MarshalIndent(memories, "", "  ")
	if err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				fmt.Sprintf("Failed to marshal results: %v", err),
			),
		)
	}

	ctx.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			string(result),
		),
	)

	return nil
}

type queryParams struct {
	keywords []string
	query    string
	cypher   string
}

func (r *Recall) initializeDatabases() error {
	if err := r.neo4j.Initialize(); err != nil {
		return fmt.Errorf("neo4j initialization: %w", err)
	}
	if err := r.qdrant.Initialize(); err != nil {
		return fmt.Errorf("qdrant initialization: %w", err)
	}
	return nil
}

func (r *Recall) prepareQueries(args map[string]any) (*queryParams, error) {
	params := &queryParams{}

	if queryRaw, ok := args["query"]; ok {
		if query, ok := queryRaw.(string); ok {
			params.query = query
		}
	}

	if cypherRaw, ok := args["cypher"]; ok {
		if cypher, ok := cypherRaw.(string); ok {
			params.cypher = cypher
		}
	}

	// Process keywords if provided
	if keywordsRaw, ok := args["keywords"]; ok {
		if keywordsStr, ok := keywordsRaw.(string); ok {
			if err := json.Unmarshal([]byte(keywordsStr), &params.keywords); err != nil {
				return nil, fmt.Errorf("keywords unmarshal: %w", err)
			}

			// Prepare semantic search query
			params.query = strings.Join(params.keywords, " ")

			// Prepare graph search query
			cypherConditions := make([]string, len(params.keywords))
			for i, keyword := range params.keywords {
				cypherConditions[i] = fmt.Sprintf("n.content CONTAINS '%s'", keyword)
			}
			params.cypher = fmt.Sprintf(`
				MATCH (n:Memory)
				WHERE %s
				RETURN n
				LIMIT 5
			`, strings.Join(cypherConditions, " OR "))
		}
	}

	return params, nil
}

func (r *Recall) executeSearches(ctx *drknow.Context, params *queryParams) (map[string]interface{}, error) {
	memories := make(map[string]interface{})

	// Execute Neo4j search
	if params.cypher != "" {
		neo4jArgs := make(map[string]any)
		if result := r.neo4j.Use(ctx.Identity.Ctx, neo4jArgs); result != "No results found" {
			var structuredMemories []map[string]interface{}
			if err := json.Unmarshal([]byte(result), &structuredMemories); err != nil {
				return nil, fmt.Errorf("neo4j results unmarshal: %w", err)
			}
			memories["structured"] = structuredMemories
		}
	}

	// Execute Qdrant search
	if params.query != "" {
		qdrantArgs := map[string]any{
			"query": params.query,
		}
		if result := r.qdrant.Use(ctx.Identity.Ctx, qdrantArgs); result != "No results found" {
			var semanticMemories []map[string]interface{}
			if err := json.Unmarshal([]byte(result), &semanticMemories); err != nil {
				return nil, fmt.Errorf("qdrant results unmarshal: %w", err)
			}
			memories["semantic"] = semanticMemories
		}
	}

	return memories, nil
}
