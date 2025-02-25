/*
Package memory provides various memory storage implementations for Caramba agents.
This package offers different memory storage solutions, from simple in-memory
stores to sophisticated vector and graph-based memory implementations that
support semantic search and complex relationships between memory items.
*/
package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
)

/*
RetrieveMemoriesByVector searches for memories using vector similarity.
It searches the vector store for memories similar to the provided query.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - query: The search query text
  - agentID: The agent ID to filter by (empty string for no filter)
  - limit: The maximum number of results to return
  - threshold: The minimum similarity score to include in results

Returns:
  - A slice of EnhancedMemoryEntry objects matching the query
  - An error if the operation fails, or nil on success
*/
func (um *UnifiedMemory) RetrieveMemoriesByVector(ctx context.Context, query string, agentID string, limit int, threshold float32) ([]EnhancedMemoryEntry, error) {
	if !um.options.EnableVectorStore || um.vectorStore == nil {
		return nil, errors.New("vector store not enabled or not available")
	}

	// Get embedding for query
	embedding, err := um.embeddingProvider.GetEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding for query: %w", err)
	}

	// Prepare filters
	filters := make(map[string]interface{})
	if agentID != "" {
		filters["agent_id"] = agentID
	}

	// Search the vector store
	results, err := um.vectorStore.Search(ctx, embedding, limit, filters)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector store: %w", err)
	}

	// Convert results to EnhancedMemoryEntry and apply threshold
	memories := make([]EnhancedMemoryEntry, 0, len(results))
	for _, result := range results {
		if result.Score < threshold {
			continue
		}

		// Extract fields from metadata
		content, _ := result.Metadata["content"].(string)
		agentID, _ := result.Metadata["agent_id"].(string)
		source, _ := result.Metadata["source"].(string)
		memType, _ := result.Metadata["type"].(string)
		createdAt, _ := result.Metadata["created_at"].(time.Time)

		// Create memory entry
		entry := EnhancedMemoryEntry{
			ID:          result.ID,
			AgentID:     agentID,
			Content:     content,
			Embedding:   result.Vector,
			Type:        MemoryType(memType),
			Source:      source,
			CreatedAt:   createdAt,
			AccessCount: 1,
			LastAccess:  time.Now(),
			Metadata:    result.Metadata,
		}

		memories = append(memories, entry)
	}

	return memories, nil
}

/*
VectorStoreProvider interface defines operations for vector-based memory storage.
This interface abstracts the vector database operations for storing and retrieving
embeddings with their associated metadata.
*/
type VectorStoreProvider interface {
	/*
		StoreVector stores a vector with the given ID and payload.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier for the vector
		  - vector: The embedding vector to store
		  - payload: Associated metadata to store with the vector

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	StoreVector(ctx context.Context, id string, vector []float32, payload map[string]interface{}) error

	/*
		Search searches for similar vectors.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - vector: The query vector to find similar vectors for
		  - limit: The maximum number of results to return
		  - filters: Optional filters to apply to the search

		Returns:
		  - A slice of SearchResult objects containing the matches
		  - An error if the operation fails, or nil on success
	*/
	Search(ctx context.Context, vector []float32, limit int, filters map[string]interface{}) ([]SearchResult, error)

	/*
		Get retrieves a specific vector by ID.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier of the vector to retrieve

		Returns:
		  - The SearchResult containing the vector and its metadata
		  - An error if the operation fails, or nil on success
	*/
	Get(ctx context.Context, id string) (*SearchResult, error)

	/*
		Delete removes a vector from the store.

		Parameters:
		  - ctx: The context for the operation, which can be used for cancellation
		  - id: The unique identifier of the vector to delete

		Returns:
		  - An error if the operation fails, or nil on success
	*/
	Delete(ctx context.Context, id string) error
}

/*
VectorEntry represents an entry in the vector store.
It holds both the original value and its vector embedding for semantic search.
*/
type VectorEntry struct {
	/* Key is the unique identifier for the entry */
	Key string
	/* Value is the stored data */
	Value interface{}
	/* Vector is the embedding representation of the content */
	Vector []float64
	/* Content is the textual representation used for searching */
	Content string
}

/*
VectorStore implements the Memory interface with a vector-based approach.
It creates embeddings for stored content and supports semantic search
based on vector similarity.
*/
type VectorStore struct {
	/* entries is the collection of vector entries */
	entries []VectorEntry
	/* llmAPI is the API endpoint used for generating embeddings */
	llmAPI string
	/* apiKey is the authentication key for the embedding service */
	apiKey string
	/* mu protects concurrent access to the entries slice */
	mu sync.RWMutex
}

/*
NewVectorStore creates a new vector store.
It initializes an empty vector store with the specified embedding API configuration.

Parameters:
  - llmAPI: The API endpoint to use for generating embeddings
  - apiKey: The authentication key for the embedding API

Returns:
  - A pointer to the initialized VectorStore
*/
func NewVectorStore(llmAPI, apiKey string) *VectorStore {
	return &VectorStore{
		entries: make([]VectorEntry, 0),
		llmAPI:  llmAPI,
		apiKey:  apiKey,
	}
}

/*
Store stores a key-value pair in memory.
It converts the value to a textual representation, generates an embedding vector,
and stores both the original value and its vector representation.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - key: The unique identifier for the value
  - value: The data to store

Returns:
  - An error if the operation fails, or nil on success
*/
func (m *VectorStore) Store(ctx context.Context, key string, value interface{}) error {
	// Get the content string representation to generate the embedding
	var content string

	switch v := value.(type) {
	case string:
		content = v
	case []byte:
		content = string(v)
	default:
		jsonData, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("failed to marshal value: %w", err)
		}
		content = string(jsonData)
	}

	// Generate embedding for the content
	vector, err := m.generateEmbedding(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	entry := VectorEntry{
		Key:     key,
		Value:   value,
		Vector:  vector,
		Content: content,
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if the key already exists and update if it does
	for i, e := range m.entries {
		if e.Key == key {
			m.entries[i] = entry
			return nil
		}
	}

	// Otherwise add a new entry
	m.entries = append(m.entries, entry)
	return nil
}

/*
Retrieve retrieves a value from memory by key.
It looks up the value associated with the provided key.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - key: The key to look up

Returns:
  - The stored value if found
  - An error if the key doesn't exist
*/
func (m *VectorStore) Retrieve(ctx context.Context, key string) (interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, entry := range m.entries {
		if entry.Key == key {
			return entry.Value, nil
		}
	}

	return nil, errors.New("key not found")
}

/*
Search searches the memory using a query.
It generates an embedding for the query and finds the most similar entries
based on vector similarity (cosine similarity).

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - query: The search query text
  - limit: The maximum number of results to return (0 means no limit)

Returns:
  - A slice of MemoryEntry objects containing the search results, sorted by relevance
  - An error if the operation fails, or nil on success
*/
func (m *VectorStore) Search(ctx context.Context, query string, limit int) ([]core.MemoryEntry, error) {
	// Generate embedding for the query
	queryVector, err := m.generateEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for query: %w", err)
	}

	type SearchResult struct {
		Entry VectorEntry
		Score float64
	}

	m.mu.RLock()

	// Calculate similarity scores
	results := make([]SearchResult, 0, len(m.entries))
	for _, entry := range m.entries {
		score := m.cosineSimilarity(queryVector, entry.Vector)
		results = append(results, SearchResult{
			Entry: entry,
			Score: score,
		})
	}

	m.mu.RUnlock()

	// Sort by similarity score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit results if needed
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}

	// Convert to MemoryEntry format
	memoryEntries := make([]core.MemoryEntry, len(results))
	for i, result := range results {
		memoryEntries[i] = core.MemoryEntry{
			Key:   result.Entry.Key,
			Value: result.Entry.Value,
			Score: result.Score,
		}
	}

	return memoryEntries, nil
}

/*
Clear clears the memory.
It removes all entries from the vector store.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation

Returns:
  - An error if the operation fails, or nil on success
*/
func (m *VectorStore) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.entries = make([]VectorEntry, 0)
	return nil
}

/*
generateEmbedding generates an embedding for the given text.
This is a simplified implementation that doesn't make actual API calls
but instead creates a simple hash-based embedding for development purposes.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - text: The text to generate an embedding for

Returns:
  - The vector embedding representation
  - An error if the operation fails, or nil on success
*/
func (m *VectorStore) generateEmbedding(ctx context.Context, text string) ([]float64, error) {
	// Don't make actual API calls in this implementation
	// This is a placeholder that should be replaced with actual embedding logic

	// For development purposes, we'll just create a very simple hash-based embedding
	// In a real implementation, you would call an embedding API
	return m.simpleHashEmbedding(text, 3072), nil
}

/*
simpleHashEmbedding creates a simple hash-based embedding for development.
This is NOT a proper embedding algorithm, just a placeholder for development.

Parameters:
  - text: The text to create an embedding for
  - dimensions: The size of the embedding vector to create

Returns:
  - A vector representation of the text with the specified dimensions
*/
func (m *VectorStore) simpleHashEmbedding(text string, dimensions int) []float64 {
	// This is NOT a proper embedding algorithm, just a placeholder
	// It creates a deterministic "embedding" based on character frequencies

	// Clean and normalize text
	text = strings.ToLower(text)

	// Initialize embedding vector
	embedding := make([]float64, dimensions)

	// Basic character frequency
	charCounts := make(map[rune]int)
	for _, char := range text {
		charCounts[char]++
	}

	// Populate embedding using character frequencies
	for char, count := range charCounts {
		pos := int(char) % dimensions
		embedding[pos] += float64(count) / float64(len(text))
	}

	// Normalize the vector
	sum := 0.0
	for _, val := range embedding {
		sum += val * val
	}

	length := 0.0
	if sum > 0 {
		length = 1.0 / (sum)
	}

	for i := range embedding {
		embedding[i] *= length
	}

	return embedding
}

/*
cosineSimilarity calculates the cosine similarity between two vectors.
It measures the cosine of the angle between two vectors, which indicates
their similarity regardless of magnitude.

Parameters:
  - a: The first vector
  - b: The second vector

Returns:
  - A similarity score between 0 and 1, where 1 indicates identical vectors
*/
func (m *VectorStore) cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / ((normA) * (normB))
}
