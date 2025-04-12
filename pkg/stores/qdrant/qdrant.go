package memory

import (
	"context"

	"slices"

	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type Document struct {
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata"`
}

type Qdrant struct {
	ctx        context.Context
	cancel     context.CancelFunc
	client     *sdk.Client
	collection string
	embedder   provider.EmbedderType
}

// QdrantOption defines a functional option pattern for Qdrant
type QdrantOption func(*Qdrant)

// WithEmbedder sets the embedder for the Qdrant instance
func WithEmbedder(embedder provider.EmbedderType) QdrantOption {
	return func(q *Qdrant) {
		q.embedder = embedder
	}
}

func NewQdrant(opts ...QdrantOption) *Qdrant {
	errnie.Debug("memory.NewQdrant")

	client, err := sdk.NewClient(&sdk.Config{
		Host:                   tweaker.GetQdrantHost(),
		Port:                   tweaker.GetQdrantPort(),
		APIKey:                 tweaker.GetQdrantAPIKey(),
		UseTLS:                 false,
		SkipCompatibilityCheck: true,
	})

	if err != nil {
		errnie.Error(err)
		return nil
	}

	// Check if collection exists, if not create it
	collections, err := client.ListCollections(context.Background())
	if err != nil {
		errnie.Error(err)
		return nil
	}

	collectionExists := slices.Contains(collections, tweaker.GetQdrantCollection())

	if !collectionExists {
		// OpenAI ada-002 embeddings are 1536-dimensional vectors
		vectorSize := uint64(tweaker.GetQdrantDimension())
		distance := sdk.Distance_Cosine

		err = client.CreateCollection(context.Background(), &sdk.CreateCollection{
			CollectionName: tweaker.GetQdrantCollection(),
			VectorsConfig: &sdk.VectorsConfig{
				Config: &sdk.VectorsConfig_Params{
					Params: &sdk.VectorParams{
						Size:     vectorSize,
						Distance: distance,
					},
				},
			},
		})
		if err != nil {
			errnie.Error(err)
			return nil
		}
	}

	ctx, cancel := context.WithCancel(context.Background())

	qdrant := &Qdrant{
		ctx:        ctx,
		cancel:     cancel,
		client:     client,
		collection: tweaker.GetQdrantCollection(),
	}

	// Apply all provided options
	for _, opt := range opts {
		opt(qdrant)
	}

	return qdrant
}

func (q *Qdrant) Get(key string) (value string, err error) {
	return "", nil
}

func (q *Qdrant) Put(key string, value string) (err error) {
	return nil
}
