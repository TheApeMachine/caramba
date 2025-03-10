package stores

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"time"

	sdk "github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
)

type QdrantData struct {
	Event *core.Event `json:"events"`
}

type Qdrant struct {
	*QdrantData
	client     *sdk.Client
	collection string
	dimensions int
	embedder   io.ReadWriteCloser

	enc *json.Encoder
	dec *json.Decoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

func NewQdrant(collection string) *Qdrant {
	errnie.Debug("NewQdrant")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	client, err := sdk.NewClient(&sdk.Config{
		Host:                   "localhost",
		Port:                   6334,
		APIKey:                 os.Getenv("QDRANT_API_KEY"),
		UseTLS:                 false,
		SkipCompatibilityCheck: true,
	})

	if err != nil {
		return nil
	}

	qdrant := &Qdrant{
		QdrantData: &QdrantData{},
		client:     client,
		collection: collection,
		dimensions: 3072,
		embedder:   provider.NewOpenAIEmbedder(os.Getenv("OPENAI_API_KEY"), "https://api.openai.com/v1"),
		enc:        json.NewEncoder(out),
		dec:        json.NewDecoder(in),
		in:         in,
		out:        out,
	}

	if err := qdrant.ensureCollection(); errnie.NewErrIO(err) != nil {
		return nil
	}

	return qdrant
}

func (qdrant *Qdrant) Read(p []byte) (n int, err error) {
	errnie.Debug("Qdrant.Read", "p", string(p))

	if qdrant.out.Len() == 0 {
		return 0, io.EOF
	}

	return qdrant.out.Read(p)
}

// executeQuery performs Qdrant queries and updates the output buffer
// This should be called from Write or other methods that modify data, not from Read
func (qdrant *Qdrant) executeQuery() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	searchedPoints, err := qdrant.client.Query(ctx, &sdk.QueryPoints{
		CollectionName: qdrant.collection,
		Query:          sdk.NewQuery(0.2, 0.1, 0.9, 0.7),
		WithPayload:    sdk.NewWithPayloadInclude("city"),
	})

	if errnie.NewErrIO(err); err != nil {
		return err
	}

	contents := make([]string, 0, len(searchedPoints))

	for _, point := range searchedPoints {
		// Check if point has content in payload
		if content, ok := point.Payload["content"]; ok {
			if contentStr := content.GetStringValue(); contentStr != "" {
				contents = append(contents, contentStr)
			}
		}
	}

	qdrant.QdrantData.Event = core.NewEvent(
		core.NewMessage("assistant", "qdrant", strings.Join(contents, "\n")),
		nil,
	)

	return errnie.NewErrIO(qdrant.enc.Encode(qdrant.QdrantData))
}

func (qdrant *Qdrant) Write(p []byte) (n int, err error) {
	errnie.Debug("Qdrant.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if qdrant.out.Len() > 0 {
		qdrant.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = qdrant.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf core.Event
	if decErr := qdrant.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		qdrant.QdrantData.Event = &buf

		// Process the data and update the output buffer
		if err := qdrant.processData(p); err != nil {
			return n, err
		}
	}

	return n, nil
}

// processData handles incoming data and populates the output buffer
func (qdrant *Qdrant) processData(p []byte) error {
	// Execute a query for reading operations
	if strings.Contains(string(p), `"operation":"search"`) {
		return qdrant.executeQuery()
	}

	// For embedding operations
	if qdrant.QdrantData.Event != nil && qdrant.QdrantData.Event.Message != nil {
		// Get the message content and send to embedder
		text := qdrant.QdrantData.Event.Message.Content
		embedderBytes, _ := json.Marshal(map[string]string{"text": text})
		if _, err := qdrant.embedder.Write(embedderBytes); err != nil {
			return errnie.NewErrIO(err)
		}

		// Read the embedding result
		var embedResult bytes.Buffer
		if _, err := io.Copy(&embedResult, qdrant.embedder); err != nil {
			return errnie.NewErrIO(err)
		}

		// Parse the embedding
		var embedding []float64
		if err := json.Unmarshal(embedResult.Bytes(), &embedding); err != nil {
			return errnie.NewErrIO(err)
		}

		// Convert float64 to float32 for SDK compatibility
		embedFloat32 := make([]float32, len(embedding))
		for i, v := range embedding {
			embedFloat32[i] = float32(v)
		}

		// Create a point with the embedding and payload
		// Using the SDK types that are imported in this file
		waitUpsert := true
		upsertPoints := []*sdk.PointStruct{
			{
				Id:      sdk.NewIDNum(uint64(time.Now().UnixNano())), // Convert to uint64
				Vectors: sdk.NewVectors(embedFloat32...),             // Use float32 values
				Payload: sdk.NewValueMap(map[string]any{
					"content": text,
				}),
			},
		}

		// Upsert the point
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		_, err := qdrant.client.Upsert(ctx, &sdk.UpsertPoints{
			CollectionName: qdrant.collection,
			Wait:           &waitUpsert,
			Points:         upsertPoints,
		})

		if err != nil {
			return errnie.NewErrIO(err)
		}
	}

	// Re-encode to the output buffer for subsequent reads
	return errnie.NewErrIO(qdrant.enc.Encode(qdrant.QdrantData))
}

func (qdrant *Qdrant) Close() (err error) {
	errnie.Debug("Qdrant.Close")

	err = qdrant.embedder.Close()

	if err != nil {
		return err
	}

	err = qdrant.client.Close()

	if err != nil {
		return err
	}

	return nil
}

func (qdrant *Qdrant) ensureCollection() error {
	errnie.Debug("Qdrant.ensureCollection")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// List all collections
	collections, err := qdrant.client.ListCollections(ctx)
	if err != nil {
		return fmt.Errorf("failed to list collections: %w", err)
	}

	// Check if our collection exists
	collectionExists := slices.Contains(collections, qdrant.collection)

	// Create collection if it doesn't exist
	if !collectionExists {
		return qdrant.createCollection()
	}

	// Collection exists
	return nil
}

func (qdrant *Qdrant) createCollection() error {
	errnie.Debug("Qdrant.createCollection")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	defaultSegmentNumber := uint64(2)

	// Create a new collection with the specified parameters
	err := qdrant.client.CreateCollection(ctx, &sdk.CreateCollection{
		CollectionName: qdrant.collection,
		VectorsConfig: sdk.NewVectorsConfig(&sdk.VectorParams{
			Size:     uint64(qdrant.dimensions),
			Distance: sdk.Distance_Dot,
		}),
		OptimizersConfig: &sdk.OptimizersConfigDiff{
			DefaultSegmentNumber: &defaultSegmentNumber,
		},
	})

	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	return nil
}
