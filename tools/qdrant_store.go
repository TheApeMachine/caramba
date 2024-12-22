package tools

import (
	"context"
	"encoding/json"
	"io"

	"github.com/invopop/jsonschema"
	"github.com/theapemachine/errnie"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

type QdrantStore struct {
	ctx        context.Context     `json:"-"`
	client     *qdrant.Store       `json:"-"`
	embedder   embeddings.Embedder `json:"-"`
	collection string              `json:"-"`
	Documents  []string            `json:"documents" jsonschema:"title=Documents,description=The text content to store,required"`
	Metadata   map[string]any      `json:"metadata" jsonschema:"title=Metadata,description=Additional context for stored documents"`
	Reasoning  string              `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of why this content should be stored"`
}

func NewQdrantStore(collection string, dimension uint64) *QdrantStore {
	// Reuse the existing connection logic from NewQdrant
	qdrantTool := NewQdrant(collection, dimension)
	return &QdrantStore{
		ctx:        qdrantTool.ctx,
		client:     qdrantTool.client,
		embedder:   qdrantTool.embedder,
		collection: collection,
	}
}

func (q *QdrantStore) GenerateSchema() string {
	schema := jsonschema.Reflect(q)
	return string(errnie.SafeMust(func() ([]byte, error) {
		return json.MarshalIndent(schema, "", "  ")
	}))
}

func (q *QdrantStore) Initialize() error {
	return nil
}

func (q *QdrantStore) Connect(rw io.ReadWriteCloser) error {
	return nil
}

func (q *QdrantStore) Use(args map[string]any) string {
	// Convert documents from []interface{} to []string
	var docs []string
	if docsInterface, ok := args["documents"].([]interface{}); ok {
		docs = make([]string, len(docsInterface))
		for i, doc := range docsInterface {
			if str, ok := doc.(string); ok {
				docs[i] = str
			}
		}
	}

	metadata, _ := args["metadata"].(map[string]any)

	if q.Reasoning != "" {
		if metadata == nil {
			metadata = make(map[string]any)
		}
		metadata["reasoning"] = q.Reasoning
	}

	for _, doc := range docs {
		errnie.SafeMust(func() ([]string, error) {
			return q.client.AddDocuments(q.ctx, []schema.Document{
				{
					PageContent: doc,
					Metadata:    metadata,
				},
			})
		})
	}

	return "memory saved in vector store"
}
