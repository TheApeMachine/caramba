package qdrant

import (
	"context"

	"slices"

	"github.com/gofiber/fiber/v3"
	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/task"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type Document struct {
	ID       string         `json:"id"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata"`
}

type QdrantQuery struct {
	Question string            `json:"question"`
	Keywords map[string]string `json:"keywords"`
}

type Qdrant struct {
	client     *sdk.Client
	collection string
	embedder   provider.EmbedderType
}

// QdrantOption defines a functional option pattern for Qdrant
type QdrantOption func(*Qdrant)

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
		errnie.New(errnie.WithError(err))
		return nil
	}

	// Check if collection exists, if not create it
	collections, err := client.ListCollections(context.Background())
	if err != nil {
		errnie.New(errnie.WithError(err))
		return nil
	}

	collectionExists := slices.Contains(collections, tweaker.GetQdrantCollection())

	if !collectionExists {
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
			errnie.New(errnie.WithError(err))
			return nil
		}
	}

	qdrant := &Qdrant{
		client:     client,
		collection: tweaker.GetQdrantCollection(),
	}

	// Apply all provided options
	for _, opt := range opts {
		opt(qdrant)
	}

	return qdrant
}

func (qdrant *Qdrant) getEmbedding(
	ctx fiber.Ctx,
	query *QdrantQuery,
) ([]float64, error) {
	embedding, err := qdrant.embedder.Embed(ctx, &task.TaskRequest{
		Params: task.NewTask(task.WithMessages(
			task.NewAssistantMessage(query.Question),
		)),
	})

	if err != nil {
		return nil, errnie.New(errnie.WithError(err))
	}

	return embedding, nil
}

func (qdrant *Qdrant) search(
	ctx fiber.Ctx,
	embedding []float64,
	filter *sdk.Filter,
) ([]*sdk.ScoredPoint, error) {
	limit := uint64(10)

	queryVector := make([]float32, len(embedding))

	for i, v := range embedding {
		queryVector[i] = float32(v)
	}

	searchedPoints, err := qdrant.client.Query(ctx.Context(), &sdk.QueryPoints{
		CollectionName: qdrant.collection,
		Limit:          &limit,
		WithPayload:    sdk.NewWithPayloadInclude("content"),
		Query:          sdk.NewQuery(queryVector...),
		Filter:         filter,
	})

	if err != nil {
		return nil, errnie.New(errnie.WithError(err))
	}

	if len(searchedPoints) == 0 {
		return nil, nil
	}

	return searchedPoints, nil
}

func (qdrant *Qdrant) Get(ctx fiber.Ctx, query *QdrantQuery) (value string, err error) {
	if query.Question == "" && len(query.Keywords) == 0 {
		return "", errnie.New(
			errnie.WithError(&MissingRequirementsError{}),
		)
	}

	if len(query.Keywords) > 0 {
		// Handle keyword-based filtering
		var should []*sdk.Condition
		for k, v := range query.Keywords {
			should = append(should, sdk.NewMatchKeyword(k, v))
		}

		filteredPoints, err := qdrant.search(ctx, nil, &sdk.Filter{
			Should: should,
		})

		if err != nil {
			return "", errnie.New(errnie.WithError(err))
		}

		if len(filteredPoints) > 0 {
			if contentVal, ok := filteredPoints[0].Payload["content"]; ok {
				if strVal, ok := contentVal.Kind.(*sdk.Value_StringValue); ok {
					return strVal.StringValue, nil
				}
			}
		}

		return "", nil
	}

	embedding, err := qdrant.getEmbedding(ctx, query)

	if err != nil {
		return "", errnie.New(errnie.WithError(err))
	}

	searchedPoints, err := qdrant.search(ctx, embedding, nil)

	if err != nil {
		return "", errnie.New(errnie.WithError(err))
	}

	// Return the content of the most similar point
	if contentVal, ok := searchedPoints[0].Payload["content"]; ok {
		if strVal, ok := contentVal.Kind.(*sdk.Value_StringValue); ok {
			return strVal.StringValue, nil
		}
	}

	return "", nil
}

func (qdrant *Qdrant) Put(ctx fiber.Ctx, docs []*Document) (err error) {
	if qdrant.embedder == nil {
		return errnie.New(errnie.WithMessage("embedder not configured"))
	}

	for _, doc := range docs {
		embedding, err := qdrant.embedder.Embed(ctx, &task.TaskRequest{
			Params: task.NewTask(task.WithMessages(
				task.NewAssistantMessage(doc.Content),
			)),
		})

		if err != nil {
			return errnie.New(errnie.WithError(err))
		}

		// Convert []float64 to []float32
		vector := make([]float32, len(embedding))

		for i, v := range embedding {
			vector[i] = float32(v)
		}

		// Create point with embedding and payload
		_, err = qdrant.client.Upsert(ctx.Context(), &sdk.UpsertPoints{
			CollectionName: qdrant.collection,
			Points: []*sdk.PointStruct{
				{
					Id:      sdk.NewID(doc.ID),
					Vectors: sdk.NewVectors(vector...),
					Payload: map[string]*sdk.Value{
						"content": {
							Kind: &sdk.Value_StringValue{
								StringValue: doc.Content,
							},
						},
					},
				},
			},
		})

		if err != nil {
			return errnie.New(errnie.WithError(err))
		}

	}

	return nil
}

// WithEmbedder sets the embedder for the Qdrant instance
func WithEmbedder(embedder provider.EmbedderType) QdrantOption {
	return func(q *Qdrant) {
		q.embedder = embedder
	}
}

type MissingRequirementsError struct{}

func (e *MissingRequirementsError) Error() string {
	return "question or keywords are required"
}
