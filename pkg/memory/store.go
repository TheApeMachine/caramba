package memory

import (
	"fmt"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stores/neo4j"
	"github.com/theapemachine/caramba/pkg/stores/qdrant"
)

// MemoryQuery represents a unified query structure for both vector and graph stores
type MemoryQuery struct {
	// Question for semantic search in vector store
	Question string `json:"question,omitempty"`
	// Cypher query for graph database
	Cypher string `json:"cypher,omitempty"`
	// Keywords for filtering in both stores
	Keywords map[string]string `json:"keywords,omitempty"`
	// Additional parameters for graph queries
	Params map[string]any `json:"params,omitempty"`
}

// Store provides a unified interface for agent memory operations
type Store struct {
	vectorStore *qdrant.Qdrant
	graphStore  *neo4j.Neo4j
	embedder    provider.EmbedderType
}

// StoreOption defines a functional option pattern for Store configuration
type StoreOption func(*Store)

// NewStore creates a new memory store instance with the provided options
func NewStore(opts ...StoreOption) *Store {
	store := &Store{}

	// Apply all provided options
	for _, opt := range opts {
		opt(store)
	}

	return store
}

// WithVectorStore sets the vector store for the memory store
func WithVectorStore(vectorStore *qdrant.Qdrant) StoreOption {
	return func(s *Store) {
		s.vectorStore = vectorStore
	}
}

// WithGraphStore sets the graph store for the memory store
func WithGraphStore(graphStore *neo4j.Neo4j) StoreOption {
	return func(s *Store) {
		s.graphStore = graphStore
	}
}

// WithEmbedder sets the embedder for vector operations
func WithEmbedder(embedder provider.EmbedderType) StoreOption {
	return func(s *Store) {
		s.embedder = embedder
	}
}

// Query performs a unified query across both vector and graph stores
func (s *Store) Query(ctx fiber.Ctx, query *MemoryQuery) (string, error) {
	var results []string

	// Query vector store if question is provided
	if query.Question != "" {
		if s.vectorStore == nil {
			return "", errnie.New(errnie.WithMessage("vector store not configured"))
		}

		vectorResult, err := s.vectorStore.Get(ctx, &qdrant.QdrantQuery{
			Question: query.Question,
			Keywords: query.Keywords,
		})
		if err != nil {
			return "", errnie.New(errnie.WithError(err))
		}
		if vectorResult != "" {
			results = append(results, fmt.Sprintf("Vector Store Results:\n%s", vectorResult))
		}
	}

	// Query graph store if cypher query is provided
	if query.Cypher != "" {
		if s.graphStore == nil {
			return "", errnie.New(errnie.WithMessage("graph store not configured"))
		}

		graphResult, err := s.graphStore.Put(ctx, neo4j.N4jQuery{
			Cypher:   query.Cypher,
			Keywords: query.Keywords,
			Params:   query.Params,
		})
		if err != nil {
			return "", errnie.New(errnie.WithError(err))
		}
		if graphResult != "" {
			results = append(results, fmt.Sprintf("Graph Store Results:\n%s", graphResult))
		}
	}

	// Return combined results
	if len(results) == 0 {
		return "No results found", nil
	}

	return fmt.Sprintf("%s\n", results), nil
}

// Store stores a document in both vector and graph stores as appropriate
func (s *Store) Store(ctx fiber.Ctx, docs []*qdrant.Document) error {
	// Store in vector store
	if s.vectorStore != nil {
		if err := s.vectorStore.Put(ctx, docs); err != nil {
			return errnie.New(errnie.WithError(err))
		}
	}

	// Store in graph store if metadata contains relationships
	if s.graphStore != nil {
		for _, doc := range docs {
			// Convert document to graph structure if it contains relationship metadata
			if relationships, ok := doc.Metadata["relationships"].(map[string]any); ok {
				cypher, params := s.buildGraphQuery(doc, relationships)
				if cypher != "" {
					_, err := s.graphStore.Put(ctx, neo4j.N4jQuery{
						Cypher: cypher,
						Params: params,
					})
					if err != nil {
						return errnie.New(errnie.WithError(err))
					}
				}
			}
		}
	}

	return nil
}

// buildGraphQuery constructs a Cypher query from document metadata
func (s *Store) buildGraphQuery(doc *qdrant.Document, relationships map[string]any) (string, map[string]any) {
	// This is a simplified example - expand based on your needs
	params := map[string]any{
		"docId":    doc.ID,
		"content":  doc.Content,
		"metadata": doc.Metadata,
	}

	// Build CREATE query for the document node
	cypher := `
		MERGE (d:Document {id: $docId})
		SET d.content = $content
		SET d.metadata = $metadata
	`

	// Add relationship creation based on metadata
	for relType, target := range relationships {
		targetID, ok := target.(string)
		if !ok {
			continue
		}

		relParam := fmt.Sprintf("rel_%s", relType)
		params[relParam] = targetID

		cypher += fmt.Sprintf(`
			MERGE (t:Document {id: $%s})
			MERGE (d)-[:%s]->(t)
		`, relParam, relType)
	}

	return cypher, params
}
