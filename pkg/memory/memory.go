// Package memory provides memory capabilities for agents
package memory

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/errnie"
)

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
		if count >= limit {
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

	errnie.Info(fmt.Sprintf("Stored memory: %s", key))
	return []core.MemoryEntry{entry}, nil
}
