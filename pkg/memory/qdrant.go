package memory

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"slices"

	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
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
	embedder   Store
}

// QdrantOption defines a functional option pattern for Qdrant
type QdrantOption func(*Qdrant)

// WithEmbedder sets the embedder for the Qdrant instance
func WithEmbedder(embedder Store) QdrantOption {
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

func (q *Qdrant) ID() string {
	return "qdrant"
}

func (q *Qdrant) Generate(
	buffer chan datura.Artifact,
	fn ...func(artifact datura.Artifact) datura.Artifact,
) chan datura.Artifact {
	errnie.Debug("memory.Qdrant.Generate")

	out := make(chan datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-q.ctx.Done():
			errnie.Debug("memory.Qdrant.Generate.ctx.Done")
			q.cancel()
			return
		case artifact := <-buffer:
			var docs []Document

			// Handle document storage
			if documents := datura.GetMetaValue[string](artifact, "documents"); documents != "" {
				if err := json.Unmarshal([]byte(documents), &docs); err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}

				points := make([]*sdk.PointStruct, 0)

				for _, doc := range docs {
					// Create artifact with document content to be embedded
					docArtifact := datura.New(
						datura.WithEncryptedPayload([]byte(doc.Content)),
					)

					// Check if embedder is available
					if q.embedder == nil {
						out <- datura.New(datura.WithError(errnie.Error(fmt.Errorf("embedder not set"))))
						return
					}

					// Get embeddings using the embedder's Generate method
					embeddingChan := q.embedder.Generate(make(chan datura.Artifact, 1))
					embeddingChan <- docArtifact
					embeddedDoc := <-embeddingChan

					// Get embeddings as bytes from the embedded document
					vectors, err := embeddedDoc.DecryptPayload()
					if err != nil {
						out <- datura.New(datura.WithError(errnie.Error(err)))
						return
					}

					// Convert bytes to float32 slice
					vectorsFloat := make([]float32, len(vectors)/4)
					for i := 0; i < len(vectors); i += 4 {
						vectorsFloat[i/4] = math.Float32frombits(binary.LittleEndian.Uint32(vectors[i : i+4]))
					}

					points = append(points, &sdk.PointStruct{
						Id:      sdk.NewIDNum(uint64(time.Now().UnixNano())),
						Vectors: sdk.NewVectors(vectorsFloat...),
						Payload: sdk.NewValueMap(map[string]any{
							"content":  doc.Content,
							"metadata": doc.Metadata,
						}),
					})
				}

				_, err := q.client.Upsert(context.Background(), &sdk.UpsertPoints{
					CollectionName: tweaker.GetQdrantCollection(),
					Points:         points,
				})

				if err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}

				out <- datura.New(datura.WithEncryptedPayload([]byte("Documents stored successfully")))
				return
			}

			// Handle search query
			if question := datura.GetMetaValue[string](artifact, "question"); question != "" {
				// Check if embedder is available
				if q.embedder == nil {
					out <- datura.New(datura.WithError(errnie.Error(fmt.Errorf("embedder not set"))))
					return
				}

				// Create artifact with question content to be embedded
				questionArtifact := datura.New(
					datura.WithEncryptedPayload([]byte(question)),
				)

				// Get embeddings using the embedder's Generate method
				embeddingChan := q.embedder.Generate(make(chan datura.Artifact, 1))
				embeddingChan <- questionArtifact
				embeddedQuestion := <-embeddingChan

				// Get embeddings as bytes from the embedded question
				vectors, err := embeddedQuestion.DecryptPayload()
				if err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}

				// Convert bytes to float32 slice
				vectorsFloat := make([]float32, len(vectors)/4)
				for i := 0; i < len(vectors); i += 4 {
					vectorsFloat[i/4] = math.Float32frombits(binary.LittleEndian.Uint32(vectors[i : i+4]))
				}

				limit := uint64(5)
				searchResult, err := q.client.Query(context.Background(), &sdk.QueryPoints{
					CollectionName: tweaker.GetQdrantCollection(),
					Query:          sdk.NewQuery(vectorsFloat...),
					Limit:          &limit,
					WithPayload: &sdk.WithPayloadSelector{
						SelectorOptions: &sdk.WithPayloadSelector_Enable{
							Enable: true,
						},
					},
				})

				if err != nil {
					out <- datura.New(datura.WithError(errnie.Error(err)))
					return
				}

				var results []map[string]any
				for _, point := range searchResult {
					results = append(results, map[string]any{
						"score":   point.Score,
						"payload": point.Payload,
					})
				}

				artifact.SetMetaValue("output", results)
				out <- artifact
				return
			}

			// If no specific operation was requested
			out <- datura.New(datura.WithError(errnie.Error(fmt.Errorf("no valid operation specified"))))
		}
	}()

	return out
}

func (q *Qdrant) Name() string {
	return "qdrant"
}
