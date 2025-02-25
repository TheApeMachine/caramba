package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// QDrantStore implements the VectorStoreProvider interface using QDrant.
type QDrantStore struct {
	// url is the connection URL for the QDrant server
	url string
	// apiKey is the authentication key for the QDrant API
	apiKey string
	// collection is the name of the collection in QDrant
	collection string
	// dimensions is the size of vectors stored in this collection
	dimensions int
	// collectionChecked tracks if we've verified the collection exists
	collectionChecked bool
}

// NewQDrantStore creates a new QDrant vector store.
//
// Parameters:
//   - url: The QDrant server URL
//   - apiKey: The QDrant API key
//   - collection: The collection name in QDrant
//   - dimensions: The dimensionality of the vectors
//
// Returns:
//   - A pointer to an initialized QDrantStore
//   - An error if initialization fails, or nil on success
func NewQDrantStore(url, apiKey, collection string, dimensions int) (*QDrantStore, error) {
	// If collection is empty, use a default name
	if collection == "" {
		collection = "points" // Default collection name
	}

	store := &QDrantStore{
		url:        url,
		apiKey:     apiKey,
		collection: collection,
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
	// Check if the collection already exists
	client := &http.Client{Timeout: 10 * time.Second}
	endpoint := fmt.Sprintf("%s/collections/%s", q.url, q.collection)

	req, err := http.NewRequest("GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to check if collection exists: %w", err)
	}
	defer resp.Body.Close()

	// If the collection doesn't exist (404), create it
	if resp.StatusCode == http.StatusNotFound {
		return q.createCollection()
	}

	// Other error statuses
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("QDrant API error checking collection: %s, status code: %d", string(body), resp.StatusCode)
	}

	// Collection exists
	return nil
}

// createCollection creates a new collection in QDrant
func (q *QDrantStore) createCollection() error {
	// Build the request to create a collection in QDrant
	type VectorParams struct {
		Size     int    `json:"size"`
		Distance string `json:"distance"`
	}

	type CreateCollectionRequest struct {
		Vectors map[string]VectorParams `json:"vectors"`
	}

	reqBody := CreateCollectionRequest{
		Vectors: map[string]VectorParams{
			"default": {
				Size:     q.dimensions,
				Distance: "Cosine", // Using cosine similarity
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal collection create request: %w", err)
	}

	// Create the HTTP request
	endpoint := fmt.Sprintf("%s/collections/%s", q.url, q.collection)
	req, err := http.NewRequest("PUT", endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}

	// Execute the request
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Check for successful response
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if strings.Contains(string(body), "already exists") {
			// Collection already exists (race condition) - this is fine
			return nil
		}
		return fmt.Errorf("QDrant API error creating collection: %s, status code: %d", string(body), resp.StatusCode)
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
	type PointStruct struct {
		ID      string                 `json:"id"`
		Vector  []float32              `json:"vector"`
		Payload map[string]interface{} `json:"payload"`
	}

	type UpsertRequest struct {
		Points []PointStruct `json:"points"`
	}

	// Build the request
	reqBody := UpsertRequest{
		Points: []PointStruct{
			{
				ID:      id,
				Vector:  vector,
				Payload: payload,
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create the HTTP request
	endpoint := fmt.Sprintf("%s/collections/%s/points", q.url, q.collection)
	req, err := http.NewRequestWithContext(ctx, "PUT", endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}

	// Execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Check for successful response
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("QDrant API error: %s, status code: %d", string(body), resp.StatusCode)
	}

	return nil
}

// Search searches for similar vectors in QDrant.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - vector: The query vector to find similar vectors for
//   - limit: The maximum number of results to return
//   - filters: Optional filters to apply to the search
//
// Returns:
//   - A slice of SearchResult objects containing the matches
//   - An error if the operation fails, or nil on success
func (q *QDrantStore) Search(ctx context.Context, vector []float32, limit int, filters map[string]interface{}) ([]SearchResult, error) {
	// Placeholder implementation
	// In a real implementation, this would construct the proper QDrant search request
	return []SearchResult{}, fmt.Errorf("QDrant search not implemented yet")
}

// Get retrieves a specific vector by ID from QDrant.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - id: The unique identifier of the vector to retrieve
//
// Returns:
//   - The SearchResult containing the vector and its metadata
//   - An error if the operation fails, or nil on success
func (q *QDrantStore) Get(ctx context.Context, id string) (*SearchResult, error) {
	// Placeholder implementation
	// In a real implementation, this would retrieve the vector from QDrant by ID
	return nil, fmt.Errorf("QDrant get by ID not implemented yet")
}

// Delete removes a vector from QDrant.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - id: The unique identifier of the vector to delete
//
// Returns:
//   - An error if the operation fails, or nil on success
func (q *QDrantStore) Delete(ctx context.Context, id string) error {
	// Placeholder implementation
	// In a real implementation, this would delete the vector from QDrant
	return fmt.Errorf("QDrant delete not implemented yet")
}
