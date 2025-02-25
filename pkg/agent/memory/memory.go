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
	"sync"

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
