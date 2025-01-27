package tools

import (
	"context"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/utils"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

type QdrantStore struct {
	ctx        context.Context     `json:"-"`
	client     *qdrant.Store       `json:"-"`
	embedder   embeddings.Embedder `json:"-"`
	collection string              `json:"-"`
	dimension  uint64              `json:"-"`
	Documents  []string            `json:"documents" jsonschema:"title=Documents,description=The text content to store,required"`
	Metadata   map[string]any      `json:"metadata" jsonschema:"title=Metadata,description=Additional context for stored documents"`
	Reasoning  string              `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of why this content should be stored"`
}

func NewQdrantStore(collection string, dimension uint64) *QdrantStore {
	return &QdrantStore{
		collection: collection,
		dimension:  dimension,
	}
}

func (q *QdrantStore) Name() string {
	return "qdrant"
}

func (q *QdrantStore) Description() string {
	return "Interact with Qdrant"
}

func (q *QdrantStore) GenerateSchema() interface{} {
	return utils.GenerateSchema[*QdrantStore]()
}

func (q *QdrantStore) Initialize() error {
	qdrantTool := NewQdrant(q.collection, q.dimension)
	if qdrantTool == nil {
		return fmt.Errorf("failed to create Qdrant tool")
	}
	if qdrantTool.client == nil {
		return fmt.Errorf("Qdrant client is nil")
	}
	if qdrantTool.embedder == nil {
		return fmt.Errorf("embedder is nil")
	}
	q.client = qdrantTool.client
	q.embedder = qdrantTool.embedder
	q.ctx = qdrantTool.ctx
	return nil
}

func (q *QdrantStore) Connect(ctx context.Context, rw io.ReadWriteCloser) error {
	return nil
}

func (q *QdrantStore) Use(ctx context.Context, args map[string]any) string {
	var (
		docs     []string
		metadata map[string]any
		ok       bool
	)

	if docsInterface, ok := args["documents"].([]interface{}); ok {
		docs = make([]string, len(docsInterface))
		for i, doc := range docsInterface {
			if str, ok := doc.(string); ok {
				docs[i] = str
			}
		}
	}

	if metadata, ok = args["metadata"].(map[string]any); ok {
		if q.Reasoning != "" {
			if metadata == nil {
				metadata = make(map[string]any)
			}
			metadata["reasoning"] = q.Reasoning
		}
	}

	for _, doc := range docs {
		_, err := q.client.AddDocuments(q.ctx, []schema.Document{
			{
				PageContent: doc,
				Metadata:    metadata,
			},
		})

		if err != nil {
			return err.Error()
		}
	}

	return "memory saved in vector store"
}
