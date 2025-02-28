/*
Package memory provides various memory storage implementations for Caramba agents.
This package offers different memory storage solutions, from simple in-memory
stores to sophisticated vector and graph-based memory implementations that
support semantic search and complex relationships between memory items.
*/
package memory

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
)

/*
InMemoryStore implements the Memory interface with an in-memory map.
It provides a simple key-value storage mechanism that persists for the
lifetime of the application but is not durable across restarts.
*/
type InMemoryStore struct {
	/* data is the underlying map storing key-value pairs */
	data map[string]interface{}
	/* mu protects concurrent access to the data map */
	mu sync.RWMutex
}

/*
NewInMemoryStore creates a new in-memory store.
It initializes an empty map for storing key-value pairs.

Returns:
  - A pointer to the initialized InMemoryStore
*/
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		data: make(map[string]interface{}),
	}
}

/*
Store stores a key-value pair in memory.
It adds or updates an entry in the in-memory store.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - key: The unique identifier for the value
  - value: The data to store

Returns:
  - An error if the operation fails, or nil on success
*/
func (m *InMemoryStore) Store(ctx context.Context, key string, value interface{}) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.data[key] = value
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
func (m *InMemoryStore) Retrieve(ctx context.Context, key string) (interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	value, exists := m.data[key]
	if !exists {
		return nil, errors.New("key not found")
	}

	return value, nil
}

/*
Search searches the memory using a query.
In this simple implementation, it returns all stored values as the
InMemoryStore doesn't support sophisticated search capabilities.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation
  - query: The search query (ignored in this implementation)
  - limit: The maximum number of results to return (0 means no limit)

Returns:
  - A slice of MemoryEntry objects containing the search results
  - An error if the operation fails, or nil on success
*/
func (m *InMemoryStore) Search(ctx context.Context, query string, limit int) ([]core.MemoryEntry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]core.MemoryEntry, 0, len(m.data))

	for k, v := range m.data {
		entry := core.MemoryEntry{
			Key:   k,
			Value: v,
			Score: 1.0, // Default score for in-memory store
		}
		result = append(result, entry)

		if limit > 0 && len(result) >= limit {
			break
		}
	}

	return result, nil
}

/*
Clear clears the memory.
It removes all key-value pairs from the in-memory store.

Parameters:
  - ctx: The context for the operation, which can be used for cancellation

Returns:
  - An error if the operation fails, or nil on success
*/
func (m *InMemoryStore) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.data = make(map[string]interface{})
	return nil
}

// BaseMemory provides a simple in-memory implementation of the Memory interface
type BaseMemory struct {
	sync.RWMutex
	memories map[string]string
}

// NewBaseMemory creates a new BaseMemory instance
func NewBaseMemory() *BaseMemory {
	return &BaseMemory{
		memories: make(map[string]string),
	}
}

// Store implements the Memory interface Store method
func (m *BaseMemory) Store(ctx context.Context, key string, value string) error {
	m.Lock()
	defer m.Unlock()

	m.memories[key] = value
	return nil
}

// Retrieve implements the Memory interface Retrieve method
func (m *BaseMemory) Retrieve(ctx context.Context, key string) (string, error) {
	m.RLock()
	defer m.RUnlock()

	if value, exists := m.memories[key]; exists {
		return value, nil
	}

	return "", fmt.Errorf("memory with key %s not found", key)
}

// Search implements the Memory interface Search method
func (m *BaseMemory) Search(ctx context.Context, query string, limit int) ([]core.MemoryEntry, error) {
	// In a real implementation, this would use embeddings and vector search
	// This is a simple implementation that just returns the most recent memories
	m.RLock()
	defer m.RUnlock()

	var entries []core.MemoryEntry

	// Get all memories and sort them by key (assuming keys include timestamps)
	var keys []string
	for key := range m.memories {
		keys = append(keys, key)
	}

	// Sort keys to get the most recent ones first (assuming keys are timestamp-based)
	sort.Sort(sort.Reverse(sort.StringSlice(keys)))

	// Take up to 'limit' entries
	count := 0
	for _, key := range keys {
		if count >= limit && limit > 0 {
			break
		}

		entries = append(entries, core.MemoryEntry{
			Key:   key,
			Value: m.memories[key],
			Score: 1.0, // Placeholder score since we're not doing real semantic search
		})

		count++
	}

	return entries, nil
}

// Clear implements the Memory interface Clear method
func (m *BaseMemory) Clear(ctx context.Context) error {
	m.Lock()
	defer m.Unlock()

	m.memories = make(map[string]string)
	return nil
}

// PrepareContext enhances the input with relevant memories
func (m *BaseMemory) PrepareContext(ctx context.Context, agentName, input string) (string, error) {
	entries, err := m.Search(ctx, input, 5) // Get 5 most relevant memories
	if err != nil {
		return input, err
	}

	if len(entries) == 0 {
		return input, nil
	}

	// Format the memories to include with the input
	memoryContext := "Relevant memories:\n"
	for _, entry := range entries {
		memoryContext += fmt.Sprintf("- %s: %s\n", entry.Key, entry.Value)
	}

	// Combine the memories with the original input
	enhancedInput := fmt.Sprintf("%s\n\nUser input: %s", memoryContext, input)
	return enhancedInput, nil
}

// ExtractMemories processes text to extract important memories
func (m *BaseMemory) ExtractMemories(ctx context.Context, agentName, text, source string) ([]core.MemoryEntry, error) {
	// In a real implementation, this would use an LLM to extract important information
	// For now, we'll just store the entire text as a single memory

	timestamp := time.Now().UnixNano()
	key := fmt.Sprintf("%s_%s_%d", agentName, source, timestamp)

	err := m.Store(ctx, key, text)
	if err != nil {
		return nil, err
	}

	entry := core.MemoryEntry{
		Key:   key,
		Value: text,
		Score: 1.0,
	}

	return []core.MemoryEntry{entry}, nil
}
