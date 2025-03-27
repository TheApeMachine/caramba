package memory

import (
	"context"
	"encoding/binary"
	"io"
	"math"
	"os"
	"strconv"

	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
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
		collection: os.Getenv("QDRANT_COLLECTION"),
		embedder:   embedder,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("memory.Qdrant.buffer")

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
				CollectionName: os.Getenv("QDRANT_COLLECTION"),
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
