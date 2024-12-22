package tools

import (
	"context"
	"encoding/json"
	"net/url"

	"github.com/gofiber/fiber/v3/client"
	"github.com/invopop/jsonschema"
	"github.com/theapemachine/errnie"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

/*
Qdrant is a wrapper around the vector store that turns it into a tool,
usable by the agent.
*/
type Qdrant struct {
	ctx        context.Context     `json:"-"`
	client     *qdrant.Store       `json:"-"`
	embedder   embeddings.Embedder `json:"-"`
	collection string              `json:"-"`
	dimension  uint64              `json:"-"`
	Operation  string              `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=add,enum=query,required"`
	SearchText string              `json:"query" jsonschema:"title=Query,description=The search text for query operation"`
	Documents  []string            `json:"documents" jsonschema:"title=Documents,description=The text content to add"`
	Metadata   map[string]any      `json:"metadata" jsonschema:"title=Metadata,description=Additional context for stored documents"`
	Reasoning  string              `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of why this content is semantically relevant"`
}

/*
Initialize initializes the Qdrant client.
*/
func (q *Qdrant) Initialize() error {
	return nil
}

/*
Connect connects to the Qdrant client.
*/
func (q *Qdrant) Connect() error {
	return nil
}

/*
Use implements the Tool interface
*/
func (qdrant *Qdrant) Use(ctx context.Context, args map[string]any) string {
	switch qdrant.Operation {
	case "add":
		if docs, ok := args["documents"].([]string); ok {
			return qdrant.Add(docs, args["metadata"].(map[string]any))
		}

		return "Invalid documents format"
	case "query":
		if query, ok := args["query"].(string); ok {
			results := errnie.SafeMust(func() ([]map[string]interface{}, error) {
				return qdrant.Query(query)
			})

			return string(errnie.SafeMust(func() ([]byte, error) {
				return json.Marshal(results)
			}))
		}

		return "Invalid query format"
	default:
		return "Unsupported operation"
	}
}

/*
GenerateSchema implements the Tool interface and renders the schema as a jsonschema string,
which can be injected into the prompt. It is used to explain to the agent how to use the tool.
*/
func (qdrant *Qdrant) GenerateSchema() string {
	schema := jsonschema.Reflect(&Qdrant{})
	return string(errnie.SafeMust(func() ([]byte, error) {
		return json.MarshalIndent(schema, "", "  ")
	}))
}

/*
NewQdrant creates a new Qdrant tool.
*/
func NewQdrant(collection string, dimension uint64) *Qdrant {
	ctx := context.Background()

	llm := errnie.SafeMust(func() (*openai.LLM, error) {
		return openai.New()
	})

	e := errnie.SafeMust(func() (embeddings.Embedder, error) {
		return embeddings.NewEmbedder(llm)
	})

	url := errnie.SafeMust(func() (*url.URL, error) {
		return url.Parse("http://localhost:6333")
	})

	createCollectionIfNotExists(collection, url, dimension)

	client := errnie.SafeMust(func() (qdrant.Store, error) {
		return qdrant.New(
			qdrant.WithURL(*url),
			qdrant.WithCollectionName(collection),
			qdrant.WithEmbedder(e),
			qdrant.WithAPIKey("qdrant-api-key"),
		)
	})

	return &Qdrant{
		ctx:        ctx,
		client:     &client,
		embedder:   e,
		collection: collection,
		dimension:  dimension,
	}
}

/*
AddDocuments is a wrapper around the qdrant.Store.AddDocuments method.
*/
func (q *Qdrant) AddDocuments(docs []schema.Document) error {
	_, err := q.client.AddDocuments(q.ctx, docs)
	return errnie.Error(err)
}

/*
SimilaritySearch is a wrapper around the qdrant.Store.SimilaritySearch method.
*/
func (q *Qdrant) SimilaritySearch(query string, k int, opts ...vectorstores.Option) ([]schema.Document, error) {
	docs, err := q.client.SimilaritySearch(q.ctx, query, k, opts...)
	return docs, errnie.Error(err)
}

/*
Query is a wrapper around the qdrant.Store.SimilaritySearch method.
*/
type QdrantResult struct {
	Metadata map[string]any `json:"metadata"`
	Content  string         `json:"content"`
}

/*
Query is a wrapper around the qdrant.Store.SimilaritySearch method.
*/
func (q *Qdrant) Query(query string) ([]map[string]interface{}, error) {
	// Perform the similarity search with the options
	docs, err := q.client.SimilaritySearch(q.ctx, query, 1, vectorstores.WithScoreThreshold(0.7))
	if errnie.Error(err) != nil {
		return nil, err
	}

	var results []map[string]interface{}

	for _, doc := range docs {
		results = append(results, map[string]interface{}{
			"metadata": doc.Metadata,
			"content":  doc.PageContent,
		})
	}

	return results, nil
}

/*
Add is a wrapper around the qdrant.Store.AddDocuments method.
*/
func (q *Qdrant) Add(docs []string, metadata map[string]any) string {
	// Add reasoning to metadata if it exists
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

/*
createCollectionIfNotExists uses an HTTP PUT call to create a collection if it does not exist.
*/
func createCollectionIfNotExists(collection string, uri *url.URL, dimension uint64) error {
	var (
		response *client.Response
		err      error
	)

	// Add API key to request headers
	headers := map[string]string{
		"Content-Type": "application/json",
	}

	// First we do a GET call to check if the collection exists
	if response, err = client.Get(uri.String()+"/collections/"+collection, client.Config{
		Header: headers,
	}); errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	if response.StatusCode() == 404 {
		// Prepare the request body for creating a new collection
		requestBody := map[string]interface{}{
			"name": collection,
			"vectors": map[string]interface{}{
				"size":     dimension,
				"distance": "Cosine",
			},
		}

		response = errnie.SafeMust(func() (*client.Response, error) {
			return client.Put(uri.String()+"/collections/"+collection, client.Config{
				Header: headers,
				Body:   requestBody,
			})
		})
	}

	errnie.Debug("collection creation response: %s", response.String())

	return nil
}
