package tools

import (
	"context"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/utils"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

type QdrantQuery struct {
	ctx        context.Context     `json:"-"`
	client     *qdrant.Store       `json:"-"`
	embedder   embeddings.Embedder `json:"-"`
	collection string              `json:"-"`
	Query      string              `json:"query" jsonschema:"title=Query,description=The search text to find similar content,required"`
	Reasoning  string              `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of why this search is relevant"`
}

func NewQdrantQuery(collection string, dimension uint64) *QdrantQuery {
	qdrantTool := NewQdrant(collection, dimension)
	return &QdrantQuery{
		ctx:        qdrantTool.ctx,
		client:     qdrantTool.client,
		embedder:   qdrantTool.embedder,
		collection: collection,
	}
}

func (q *QdrantQuery) Name() string {
	return "qdrant"
}

func (q *QdrantQuery) Description() string {
	return "Interact with Qdrant"
}

func (q *QdrantQuery) GenerateSchema() interface{} {
	return utils.GenerateSchema[*QdrantQuery]()
}

func (q *QdrantQuery) Initialize() error {
	return nil
}

func (q *QdrantQuery) Connect(ctx context.Context, rw io.ReadWriteCloser) error {
	return nil
}

func (q *QdrantQuery) Use(ctx context.Context, args map[string]any) string {
	var (
		query   string
		ok      bool
		buf     []byte
		err     error
		results []map[string]interface{}
	)

	if query, ok = args["query"].(string); !ok {
		return "query is required"
	}

	docs, err := q.client.SimilaritySearch(q.ctx, query, 1, vectorstores.WithScoreThreshold(0.7))
	if err != nil {
		return err.Error()
	}

	var out []map[string]interface{}
	for _, doc := range docs {
		out = append(out, map[string]interface{}{
			"metadata": doc.Metadata,
			"content":  doc.PageContent,
		})
	}

	if len(out) == 0 {
		return "No results found"
	}

	if buf, err = json.Marshal(results); err != nil {
		return err.Error()
	}

	return string(buf)
}
