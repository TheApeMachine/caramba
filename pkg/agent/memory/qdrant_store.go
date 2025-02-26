package memory

import (
	"context"
	"fmt"
	"os"
	"time"

	"slices"

	"github.com/qdrant/go-client/qdrant"
)

//------------------------------------------------------------------------------
// QDrant Base Configuration
//------------------------------------------------------------------------------

// QDrantConfig holds configuration for the QDrant client
type QDrantConfig struct {
	Host           string
	Port           int
	APIKey         string
	UseTLS         bool
	CollectionName string
}

//------------------------------------------------------------------------------
// QDrant Vector Store Provider
//------------------------------------------------------------------------------

// QDrantStore implements the VectorStoreProvider interface using QDrant.
type QDrantStore struct {
	// client is the QDrant client
	client *qdrant.Client
	// collection is the name of the collection in QDrant
	collection string
	// dimensions is the size of vectors stored in this collection
	dimensions int
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
func NewQDrantStore(urlStr, apiKey, collection string, dimensions int) (*QDrantStore, error) {
	// Ensure dimensions match the model being used
	if dimensions <= 0 {
		dimensions = 3072 // Default for OpenAI text-embedding-3-large embeddings
	}

	// Create the client with simple configuration
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:                   "localhost",
		Port:                   6334,
		APIKey:                 os.Getenv("QDRANT_API_KEY"),
		UseTLS:                 false,
		SkipCompatibilityCheck: true,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to connect to QDrant: %w", err)
	}

	store := &QDrantStore{
		client:     client,
		collection: "long-term-memory",
		dimensions: dimensions,
	}

	// Ensure the collection exists
	if err := store.ensureCollection(); err != nil {
		return nil, fmt.Errorf("failed to ensure collection exists: %w", err)
	}

	return store, nil
}

// ensureCollection checks if the collection exists and creates it if it doesn't
func (q *QDrantStore) ensureCollection() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// List all collections
	collections, err := q.client.ListCollections(ctx)
	if err != nil {
		return fmt.Errorf("failed to list collections: %w", err)
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
		return fmt.Errorf("failed to create collection: %w", err)
	}

	return nil
}

// StoreVector stores a vector with the given ID and payload in QDrant.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - id: The unique identifier for the vector
//   - vector: The embedding vector to store
//   - payload: Associated metadata to store with the vector
//
// Returns:
//   - An error if the operation fails, or nil on success
func (q *QDrantStore) StoreVector(ctx context.Context, id string, vector []float32, payload map[string]interface{}) error {
	// Preprocess payload to handle time.Time values and map[string]string values
	processedPayload := make(map[string]interface{})
	for k, v := range payload {
		// Convert time.Time to RFC3339 string format
		if timeVal, ok := v.(time.Time); ok {
			processedPayload[k] = timeVal.Format(time.RFC3339)
		} else if strMap, ok := v.(map[string]string); ok {
			// Convert map[string]string to map[string]interface{}
			interfaceMap := make(map[string]interface{})
			for sk, sv := range strMap {
				interfaceMap[sk] = sv
			}
			processedPayload[k] = interfaceMap
		} else {
			processedPayload[k] = v
		}
	}

	// Convert payload to the Qdrant Value map
	qdrantPayload := qdrant.NewValueMap(processedPayload)

	// Create point with vectors
	points := []*qdrant.PointStruct{
		{
			Id:      qdrant.NewIDUUID(id),
			Vectors: qdrant.NewVectors(vector...),
			Payload: qdrantPayload,
		},
	}

	// Upsert the point
	_, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: q.collection,
		Points:         points,
	})

	if err != nil {
		return fmt.Errorf("failed to store vector in QDrant: %w", err)
	}

	return nil
}

// Search searches for vectors similar to the given query in the QDrant store.
func (q *QDrantStore) Search(
	ctx context.Context,
	vector []float32,
	limit int,
	filters map[string]interface{},
) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 10
	}

	// Validate vector dimensions
	if len(vector) != q.dimensions {
		return nil, fmt.Errorf("vector dimension mismatch: expected %d, got %d", q.dimensions, len(vector))
	}

	limitUint := uint64(limit)

	// Create query parameters
	queryParams := &qdrant.QueryPoints{
		CollectionName: q.collection,
		Query:          qdrant.NewQuery(vector...),
		Limit:          &limitUint,
		WithPayload:    qdrant.NewWithPayload(true),
	}

	// Execute the search
	searchedPoints, err := q.client.Query(ctx, queryParams)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Convert results
	results := make([]SearchResult, 0, len(searchedPoints))
	for _, point := range searchedPoints {
		// Get ID
		var id string
		if pointID := point.Id; pointID != nil {
			if uuidID := pointID.GetUuid(); uuidID != "" {
				id = uuidID
			} else if numID := pointID.GetNum(); numID != 0 {
				id = fmt.Sprintf("%d", numID)
			}
		}

		// Extract metadata
		metadata := make(map[string]string)
		for k, v := range point.Payload {
			metadata[k] = v.GetStringValue()
		}

		// Add to results
		results = append(results, SearchResult{
			ID:       id,
			Score:    float32(point.Score),
			Metadata: metadata,
		})
	}

	return results, nil
}

// Delete removes a vector from QDrant.
func (q *QDrantStore) Delete(ctx context.Context, id string) error {
	// Delete the point
	_, err := q.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: q.collection,
		Points:         qdrant.NewPointsSelector(qdrant.NewIDUUID(id)),
	})

	if err != nil {
		return fmt.Errorf("failed to delete vector: %w", err)
	}

	return nil
}

// Get retrieves a specific vector by ID from QDrant.
func (q *QDrantStore) Get(ctx context.Context, id string) (*SearchResult, error) {
	// Retrieve the point by ID
	points, err := q.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: q.collection,
		Ids:            []*qdrant.PointId{qdrant.NewIDUUID(id)},
		WithPayload:    qdrant.NewWithPayload(true),
	})

	if err != nil {
		return nil, fmt.Errorf("failed to get vector: %w", err)
	}

	if len(points) == 0 {
		return nil, fmt.Errorf("vector with ID %s not found", id)
	}

	point := points[0]

	// Extract metadata
	metadata := make(map[string]string)
	for k, v := range point.Payload {
		metadata[k] = v.GetStringValue()
	}

	return &SearchResult{
		ID:       id,
		Score:    1.0, // Default score for direct retrieval
		Metadata: metadata,
	}, nil
}
