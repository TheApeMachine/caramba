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

	"github.com/qdrant/go-client/qdrant"
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
	client     *qdrant.Client
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

	client, err := qdrant.NewClient(&qdrant.Config{
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
	errnie.Debug("Qdrant.Read")

	if qdrant.out.Len() == 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		searchedPoints, err := qdrant.client.Query(ctx, &sdk.QueryPoints{
			CollectionName: qdrant.collection,
			Query:          sdk.NewQuery(0.2, 0.1, 0.9, 0.7),
			WithPayload:    sdk.NewWithPayloadInclude("city"),
		})

		if errnie.NewErrIO(err); err != nil {
			return 0, err
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

		if err = errnie.NewErrIO(qdrant.enc.Encode(qdrant.QdrantData)); err != nil {
			return 0, err
		}
	}

	return qdrant.out.Read(p)
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
	var buf QdrantData
	if decErr := qdrant.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		qdrant.QdrantData.Event = buf.Event

		// Re-encode to the output buffer for subsequent reads
		if encErr := qdrant.enc.Encode(qdrant.QdrantData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	waitUpsert := true
	upsertPoints := []*sdk.PointStruct{
		{
			Id:      sdk.NewIDNum(1),
			Vectors: sdk.NewVectors(0.05, 0.61, 0.76, 0.74),
			Payload: sdk.NewValueMap(map[string]any{
				"city":    "Berlin",
				"country": "Germany",
				"count":   1000000,
				"square":  12.5,
			}),
		},
	}

	_, err = qdrant.client.Upsert(ctx, &sdk.UpsertPoints{
		CollectionName: qdrant.collection,
		Wait:           &waitUpsert,
		Points:         upsertPoints,
	})

	return n, nil
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
