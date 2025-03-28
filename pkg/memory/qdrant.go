package memory

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"time"

	"slices"

	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Document struct {
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata"`
}

type Qdrant struct {
	client     *sdk.Client
	collection string
	buffer     *stream.Buffer
	embedder   *provider.OpenAIEmbedder
}

func NewQdrant() *Qdrant {
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

	embedder := provider.NewOpenAIEmbedder()

	qdrant := &Qdrant{
		client:     client,
		collection: tweaker.GetQdrantCollection(),
		embedder:   embedder,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("memory.Qdrant.buffer")

			var docs []Document

			// Handle document storage
			if documents := datura.GetMetaValue[string](artifact, "documents"); documents != "" {
				if err = json.Unmarshal([]byte(documents), &docs); err != nil {
					return errnie.Error(err)
				}

				points := make([]*sdk.PointStruct, 0)

				for _, doc := range docs {
					docArtifact := datura.New(
						datura.WithPayload([]byte(doc.Content)),
					)

					if err = workflow.NewFlipFlop(docArtifact, embedder); err != nil {
						return errnie.Error(err)
					}

					// Get embeddings directly as float32 slice
					vectors, err := docArtifact.DecryptPayload()
					if err != nil {
						return errnie.Error(err)
					}

					vectorsFloat := make([]float32, len(vectors)/4)
					for i := 0; i < len(vectors); i += 4 {
						vectorsFloat[i/4] = math.Float32frombits(binary.LittleEndian.Uint32(vectors[i : i+4]))
					}

					points = append(points, &sdk.PointStruct{
						Id:      sdk.NewIDNum(uint64(time.Now().UnixNano())),
						Vectors: sdk.NewVectors(vectorsFloat...),
						Payload: sdk.NewValueMap(map[string]any{
							"content": doc.Content,
						}),
					})
				}

				_, err = client.Upsert(context.Background(), &sdk.UpsertPoints{
					CollectionName: tweaker.GetQdrantCollection(),
					Points:         points,
				})
				return errnie.Error(err)
			}

			// Handle search query
			if question := datura.GetMetaValue[string](artifact, "question"); question != "" {
				if err = workflow.NewFlipFlop(artifact, embedder); err != nil {
					return errnie.Error(err)
				}

				// Get embeddings directly as float32 slice
				vectors := datura.GetMetaValue[[]float32](artifact, "embeddings")

				limit := uint64(5)
				searchResult, err := client.Query(context.Background(), &sdk.QueryPoints{
					CollectionName: tweaker.GetQdrantCollection(),
					Query:          sdk.NewQuery(vectors...),
					Limit:          &limit,
					WithPayload: &sdk.WithPayloadSelector{
						SelectorOptions: &sdk.WithPayloadSelector_Enable{
							Enable: true,
						},
					},
				})
				if err != nil {
					return errnie.Error(err)
				}

				var results []map[string]any
				for _, point := range searchResult {
					results = append(results, map[string]any{
						"score":   point.Score,
						"payload": point.Payload,
					})
				}

				artifact.SetMetaValue("output", results)
			}

			return nil
		}),
	}

	return qdrant
}

func (q *Qdrant) Read(p []byte) (n int, err error) {
	errnie.Debug("Qdrant.Read")
	if q.buffer == nil {
		return 0, errnie.Error(err)
	}
	return q.buffer.Read(p)
}

func (q *Qdrant) Write(p []byte) (n int, err error) {
	errnie.Debug("Qdrant.Write")
	if q.buffer == nil {
		return 0, errnie.Error(err)
	}
	return q.buffer.Write(p)
}

func (q *Qdrant) Close() error {
	errnie.Debug("Qdrant.Close")
	if q.buffer != nil {
		return q.buffer.Close()
	}
	return nil
}
