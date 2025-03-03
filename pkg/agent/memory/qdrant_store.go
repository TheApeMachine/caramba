package memory

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"slices"

	"github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// EmbeddingProvider defines the interface for embedding text
type EmbeddingProvider interface {
	GetEmbedding(ctx context.Context, text string) ([]float32, error)
}

// SearchResult represents a search result from the vector store
type SearchResult struct {
	ID       string
	Score    float32
	Metadata map[string]string
}

// QDrantStore implements the VectorStoreProvider interface using QDrant.
type QDrantStore struct {
	hub        *hub.Queue
	logger     *output.Logger
	client     *qdrant.Client
	collection string
	dimensions int
	embedder   EmbeddingProvider
}

// NewQDrantStore creates a new QDrant vector store.
//
// Parameters:
//   - urlStr: The QDrant server URL (format: host:port)
//   - apiKey: The QDrant API key
//   - collection: The collection name in QDrant
//   - dimensions: The dimensionality of the vectors
//
// Returns:
//   - A pointer to an initialized QDrantStore
//   - An error if initialization fails, or nil on success
func NewQDrantStore(collection string, embedder EmbeddingProvider) *QDrantStore {
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

	store := &QDrantStore{
		hub:        hub.NewQueue(),
		logger:     output.NewLogger(),
		client:     client,
		collection: collection,
		dimensions: 3072,
		embedder:   embedder,
	}

	// Ensure the collection exists
	if err := store.ensureCollection(); err != nil {
		store.logger.Error("qdrant", err)
		return nil
	}

	store.logger.Success("qdrant", "online")
	store.hub.Add(&hub.Event{
		Origin:  "qdrant",
		Topic:   hub.TopicTypeStore,
		Type:    hub.EventTypeStatus,
		Message: "online",
	})

	return store
}

// Query searches for a string in the vector store
func (q *QDrantStore) Query(ctx context.Context, queryParams map[string]any) (string, error) {
	// Extract the query string from the parameters
	queryStr, ok := queryParams["query"].(string)
	if !ok || queryStr == "" {
		return "", q.logger.Error(
			"qdrant",
			fmt.Errorf("query parameter must contain a non-empty 'query' string field"),
		)
	}

	// Use the embedder to convert the query to a vector
	vector, err := q.embedder.GetEmbedding(ctx, queryStr)
	if err != nil {
		return "", q.logger.Error(
			"qdrant",
			fmt.Errorf("failed to embed query: %w", err),
		)
	}

	// Extract limit if provided, or use default
	limit := 10
	if limitVal, ok := queryParams["limit"].(int); ok && limitVal > 0 {
		limit = limitVal
	}

	results, err := q.Search(ctx, vector, limit)
	if err != nil {
		return "", q.logger.Error(
			"qdrant",
			fmt.Errorf("failed to search: %v", err),
		)
	}

	if len(results) == 0 {
		return "", q.logger.Error(
			"qdrant",
			fmt.Errorf("no results found for query: %s", queryStr),
		)
	}

	var out strings.Builder

	for _, result := range results {
		out.WriteString(result)
		q.hub.Add(&hub.Event{
			Origin:  "qdrant",
			Topic:   hub.TopicTypeStore,
			Type:    hub.EventTypeQuery,
			Message: result,
		})
	}

	return out.String(), nil
}

// Mutate stores data in the vector store
func (q *QDrantStore) Mutate(ctx context.Context, payload map[string]any) error {
	// Extract ID if provided, otherwise generate one
	var id string
	if idVal, ok := payload["id"]; ok {
		if idStr, ok := idVal.(string); ok {
			id = idStr
		}
	}

	// If no ID was provided or it wasn't a string, generate a new one
	if id == "" {
		id = fmt.Sprintf("%d", time.Now().UnixNano())
	}

	// Extract content to embed
	var content string
	if contentVal, ok := payload["content"]; ok {
		if contentStr, ok := contentVal.(string); ok {
			content = contentStr
		}
	}

	if content == "" {
		return q.logger.Error(
			"qdrant",
			fmt.Errorf("payload must contain a non-empty 'content' field"),
		)
	}

	// Use the embedder to generate the vector
	vector, err := q.embedder.GetEmbedding(ctx, content)
	if err != nil {
		return q.logger.Error(
			"qdrant",
			fmt.Errorf("failed to embed content: %w", err),
		)
	}

	// Store the vector and payload
	return q.StoreVector(ctx, id, vector, payload)
}

// ensureCollection checks if the collection exists and creates it if it doesn't
func (q *QDrantStore) ensureCollection() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// List all collections
	collections, err := q.client.ListCollections(ctx)
	if err != nil {
		return q.logger.Error(
			"qdrant",
			fmt.Errorf("failed to list collections: %w", err),
		)
	}

	// Check if our collection exists
	collectionExists := slices.Contains(collections, q.collection)

	// Create collection if it doesn't exist
	if !collectionExists {
		return q.createCollection()
	}

	// Collection exists
	return nil
}

// createCollection creates a new collection in QDrant
func (q *QDrantStore) createCollection() error {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	defaultSegmentNumber := uint64(2)

	// Create a new collection with the specified parameters
	err := q.client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: q.collection,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     uint64(q.dimensions),
			Distance: qdrant.Distance_Cosine,
		}),
		OptimizersConfig: &qdrant.OptimizersConfigDiff{
			DefaultSegmentNumber: &defaultSegmentNumber,
		},
	})

	if err != nil {
		return q.logger.Error(
			"qdrant",
			fmt.Errorf("failed to create collection: %w", err),
		)
	}

	return nil
}

// StoreVector stores a vector with the given ID and payload in QDrant.
func (q *QDrantStore) StoreVector(ctx context.Context, id string, vector []float32, payload map[string]interface{}) error {
	// Validate inputs
	if id == "" {
		return q.logger.Error(
			"qdrant",
			fmt.Errorf("vector ID cannot be empty"),
		)
	}

	if len(vector) != q.dimensions {
		return q.logger.Error(
			"qdrant",
			fmt.Errorf("vector dimension mismatch: expected %d, got %d", q.dimensions, len(vector)),
		)
	}

	// Create point with vectors
	points := []*qdrant.PointStruct{
		{
			Id:      qdrant.NewIDUUID(id),
			Vectors: qdrant.NewVectors(vector...),
			Payload: qdrant.NewValueMap(payload),
		},
	}

	// Use wait flag to ensure points are available immediately after upsert
	waitUpsert := true

	// Upsert the point
	_, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: q.collection,
		Points:         points,
		Wait:           &waitUpsert,
	})

	if err != nil {
		return q.logger.Error(
			"qdrant",
			fmt.Errorf("failed to store vector in QDrant: %w", err),
		)
	}

	return nil
}

// Search searches for vectors similar to the given query in the QDrant store.
func (q *QDrantStore) Search(
	ctx context.Context,
	vector []float32,
	limit int,
) ([]string, error) {
	if limit <= 0 {
		limit = 10
	}

	// Validate vector dimensions
	if len(vector) != q.dimensions {
		return nil, q.logger.Error(
			"qdrant",
			fmt.Errorf("vector dimension mismatch: expected %d, got %d", q.dimensions, len(vector)),
		)
	}

	limitUint := uint64(limit)

	// Create query using the helper function
	queryParams := &qdrant.QueryPoints{
		CollectionName: q.collection,
		Query:          qdrant.NewQuery(vector...),
		Limit:          &limitUint,
		WithPayload:    qdrant.NewWithPayload(true),
	}

	// Execute the search
	searchedPoints, err := q.client.Query(ctx, queryParams)
	if err != nil {
		return nil, q.logger.Error(
			"qdrant",
			fmt.Errorf("search failed: %w", err),
		)
	}

	// Extract content strings from search results
	contents := make([]string, 0, len(searchedPoints))
	for _, point := range searchedPoints {
		// Check if point has content in payload
		if content, ok := point.Payload["content"]; ok {
			if contentStr := content.GetStringValue(); contentStr != "" {
				contents = append(contents, contentStr)
			}
		}
	}

	return contents, nil
}
