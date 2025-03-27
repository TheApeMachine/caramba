package memory

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"strconv"
	"time"

	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type Qdrant struct {
	client     *sdk.Client
	collection string
	buffer     *stream.Buffer
	embedder   io.ReadWriteCloser
}

func NewQdrant() *Qdrant {
	errnie.Debug("memory.NewQdrant")

	port, err := strconv.Atoi(os.Getenv("QDRANT_PORT"))
	if errnie.Error(err) != nil {
		return nil
	}

	client, err := sdk.NewClient(&sdk.Config{
		Host:   os.Getenv("QDRANT_HOST"),
		Port:   port,
		APIKey: os.Getenv("QDRANT_API_KEY"),
	})

	if err != nil {
		errnie.Error(err)
		return nil
	}

	// Initialize embedder based on configuration
	// Default to OpenAI embedder if not specified
	embedder := provider.NewOpenAIEmbedder(os.Getenv("OPENAI_API_KEY"), "")

	qdrant := &Qdrant{
		client:     client,
		collection: tweaker.GetQdrantCollection(),
		embedder:   embedder,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("memory.Qdrant.buffer")

			// Check for documents to write
			if documents := datura.GetMetaValue[string](artifact, "documents"); documents != "" {
				var docs []struct {
					Content  string                 `json:"content"`
					Metadata map[string]interface{} `json:"metadata"`
				}
				if err := json.Unmarshal([]byte(documents), &docs); err != nil {
					return errnie.Error(err)
				}

				points := make([]*sdk.PointStruct, len(docs))
				for i, doc := range docs {
					// Generate embeddings for the document content
					if _, err = embedder.Write([]byte(doc.Content)); err != nil {
						return errnie.Error(err)
					}

					// Read the embeddings
					embeddings := make([]byte, 1024*1024)
					n, err := embedder.Read(embeddings)
					if err != nil && err != io.EOF {
						return errnie.Error(err)
					}

					// Convert embeddings bytes to float32 slice
					vectors := make([]float32, n/4)
					for i := 0; i < n; i += 4 {
						bits := binary.LittleEndian.Uint32(embeddings[i : i+4])
						vectors[i/4] = math.Float32frombits(bits)
					}

					// Create point with unique ID and metadata
					points[i] = &sdk.PointStruct{
						Id:      sdk.NewIDNum(uint64(time.Now().UnixNano())),
						Vectors: sdk.NewVectors(vectors...),
						Payload: sdk.NewValueMap(doc.Metadata),
					}
				}

				// Store the points in Qdrant
				_, err = client.Upsert(context.Background(), &sdk.UpsertPoints{
					CollectionName: tweaker.GetQdrantCollection(),
					Points:         points,
				})
				if err != nil {
					return errnie.Error(err)
				}

				// Return early since this was a write operation
				return nil
			}

			// Get the question from the artifact metadata
			question := datura.GetMetaValue[string](artifact, "question")
			if question == "" {
				return nil
			}

			// Write the question to the embedder to get embeddings
			if _, err = embedder.Write([]byte(question)); err != nil {
				return errnie.Error(err)
			}

			// Read the embeddings from the embedder
			embeddings := make([]byte, 1024*1024) // Adjust buffer size as needed
			n, err := embedder.Read(embeddings)
			if err != nil && err != io.EOF {
				return errnie.Error(err)
			}

			// Convert embeddings bytes directly to float32 slice
			vectors := make([]float32, n/4)
			for i := 0; i < n; i += 4 {
				bits := binary.LittleEndian.Uint32(embeddings[i : i+4])
				vectors[i/4] = math.Float32frombits(bits)
			}

			// Perform semantic search in Qdrant
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
				WithVectors: &sdk.WithVectorsSelector{
					SelectorOptions: &sdk.WithVectorsSelector_Enable{
						Enable: true,
					},
				},
			})

			if err != nil {
				return errnie.Error(err)
			}

			// Process search results
			var results []map[string]any
			for _, point := range searchResult {
				result := map[string]any{
					"score":   point.Score,
					"payload": point.Payload,
				}
				results = append(results, result)
			}

			// Store results in artifact metadata
			artifact.SetMetaValue("output", results)

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
