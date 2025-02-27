/*
Package memory provides memory-related functionality for the agent system.
This file contains implementations for batch memory operations that
optimize performance when working with multiple memories at once.
*/
package memory

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/output"
)

// BatchMemoryItem represents a single memory entry for batch processing
type BatchMemoryItem struct {
	// Content is the actual memory content
	Content string
	// AgentID is the ID of the agent who owns the memory (empty for global)
	AgentID string
	// Type indicates the type of memory (personal/global)
	Type MemoryType
	// Source identifies the origin of the memory
	Source string
	// Metadata stores additional information about the memory
	Metadata map[string]string
}

// DefaultBatchMemoryItem creates a default batch memory item
func DefaultBatchMemoryItem(content string) BatchMemoryItem {
	return BatchMemoryItem{
		Content:  content,
		AgentID:  "",
		Type:     MemoryTypeGlobal,
		Source:   "batch_operation",
		Metadata: make(map[string]string),
	}
}

// BatchEmbeddingProvider defines the interface for embedding providers
// that support efficient batch operations
type BatchEmbeddingProvider interface {
	// GetEmbedding retrieves the embedding for a single text
	GetEmbedding(ctx context.Context, text string) ([]float32, error)

	// GetBatchEmbeddings retrieves embeddings for multiple texts in a single operation
	GetBatchEmbeddings(ctx context.Context, texts []string) ([][]float32, error)
}

// BatchMemoryProcessor provides batch operations for memory handling
type BatchMemoryProcessor struct {
	// memory is the reference to the unified memory system
	memory *UnifiedMemory
}

// NewBatchMemoryProcessor creates a new batch memory processor
func NewBatchMemoryProcessor(memory *UnifiedMemory) *BatchMemoryProcessor {
	return &BatchMemoryProcessor{
		memory: memory,
	}
}

// BatchVectorStore defines an interface for vector stores that support batch operations
type BatchVectorStore interface {
	VectorStoreProvider

	// BatchStoreVectors stores multiple vectors in a single operation
	BatchStoreVectors(ctx context.Context, ids []string, vectors [][]float32, payloads []map[string]interface{}) error

	// BatchSearch performs multiple searches efficiently
	BatchSearch(ctx context.Context, vectors [][]float32, limit int, filters []map[string]interface{}) ([][]SearchResult, error)
}

// BatchStoreMemories stores multiple memories efficiently in a single operation
func (b *BatchMemoryProcessor) BatchStoreMemories(
	ctx context.Context,
	items []BatchMemoryItem,
) ([]string, error) {
	if len(items) == 0 {
		return nil, nil
	}

	output.Action("memory", "batch_store", fmt.Sprintf("Batch storing %d memories", len(items)))

	// Step 1: Get embeddings for all contents at once if possible
	embeddings, err := b.getBatchEmbeddings(ctx, items)
	if err != nil {
		output.Warn(fmt.Sprintf("Batch embedding generation failed: %v, falling back to sequential", err))
	}

	// Step 2: Prepare memory entries and vector batch data
	memoryIDs := make([]string, len(items))
	memoryEntries := make([]*EnhancedMemoryEntry, len(items))

	// For the new batch interface
	vectorIDs := make([]string, 0, len(items))
	vectorsList := make([][]float32, 0, len(items))
	payloadsList := make([]map[string]interface{}, 0, len(items))

	for i, item := range items {
		// Generate a unique ID
		memoryID := uuid.New().String()
		memoryIDs[i] = memoryID

		// Prepare the memory entry
		entry := &EnhancedMemoryEntry{
			ID:              memoryID,
			AgentID:         item.AgentID,
			Content:         item.Content,
			Type:            item.Type,
			Source:          item.Source,
			CreatedAt:       time.Now(),
			AccessCount:     0,
			LastAccess:      time.Now(),
			Metadata:        item.Metadata,
			ImportanceScore: 0.5, // Default initial importance
			RelevanceCache:  NewRelevanceCache(),
			UsageStats:      NewMemoryUsageStats(),
		}

		// Add embedding if available
		if i < len(embeddings) && embeddings[i] != nil {
			entry.Embedding = embeddings[i]

			// Prepare vector data for batch storage
			vectorIDs = append(vectorIDs, memoryID)
			vectorsList = append(vectorsList, embeddings[i])

			// Prepare payload
			payload := map[string]interface{}{
				"agent_id":   item.AgentID,
				"content":    item.Content,
				"type":       string(item.Type),
				"source":     item.Source,
				"created_at": entry.CreatedAt,
			}

			// Add metadata as individual fields
			for k, v := range item.Metadata {
				payload[k] = v
			}

			payloadsList = append(payloadsList, payload)
		}

		memoryEntries[i] = entry
	}

	// Step 3: Store all entries in memory
	b.memory.mutex.Lock()
	for _, entry := range memoryEntries {
		b.memory.memoryData[entry.ID] = entry
	}
	b.memory.mutex.Unlock()

	// Step 4: Store in vector store (batch operation if supported)
	if b.memory.options.EnableVectorStore && b.memory.vectorStore != nil && len(vectorIDs) > 0 {
		vectorStoreSpinner := output.StartSpinner("Batch storing in vector database")

		if batchStore, ok := b.memory.vectorStore.(BatchVectorStore); ok {
			err := batchStore.BatchStoreVectors(ctx, vectorIDs, vectorsList, payloadsList)
			if err != nil {
				output.StopSpinner(vectorStoreSpinner, "")
				output.Error("Failed to batch store in vector store", err)
			} else {
				output.StopSpinner(vectorStoreSpinner, fmt.Sprintf("Stored %d memories in vector database", len(vectorIDs)))
			}
		} else {
			// Fallback to individual storage
			output.StopSpinner(vectorStoreSpinner, "")
			output.Warn("Batch storage not supported by vector store implementation")

			var wg sync.WaitGroup
			errChan := make(chan error, len(vectorIDs))

			for i, id := range vectorIDs {
				wg.Add(1)
				go func(memID string, vec []float32, payload map[string]interface{}) {
					defer wg.Done()
					if err := b.memory.vectorStore.StoreVector(ctx, memID, vec, payload); err != nil {
						errChan <- fmt.Errorf("failed to store memory %s: %w", memID, err)
					}
				}(id, vectorsList[i], payloadsList[i])
			}

			wg.Wait()
			close(errChan)

			// Check if any errors occurred
			select {
			case err := <-errChan:
				output.Error("Error during fallback individual vector storage", err)
			default:
				// No errors
			}
		}
	}

	// Step 5: Store in graph store (could be extended for batch operations)
	if b.memory.options.EnableGraphStore && b.memory.graphStore != nil {
		graphStoreSpinner := output.StartSpinner("Storing in graph database")

		// Currently, we call individual operations
		// A future optimization could implement a batch interface for graph stores
		for _, entry := range memoryEntries {
			properties := map[string]interface{}{
				"agent_id":   entry.AgentID,
				"content":    entry.Content,
				"type":       string(entry.Type),
				"source":     entry.Source,
				"created_at": entry.CreatedAt.Format(time.RFC3339),
			}

			for k, v := range entry.Metadata {
				properties[k] = v
			}

			labels := []string{"Memory"}
			if entry.Type == MemoryTypePersonal {
				labels = append(labels, "Personal")
			} else {
				labels = append(labels, "Global")
			}

			err := b.memory.graphStore.CreateNode(ctx, entry.ID, labels, properties)
			if err != nil {
				output.Error(fmt.Sprintf("Failed to store memory %s in graph store", entry.ID), err)
			}
		}

		output.StopSpinner(graphStoreSpinner, fmt.Sprintf("Stored %d memories in graph database", len(memoryEntries)))
	}

	output.Result(fmt.Sprintf("Successfully batch stored %d memories", len(items)))
	return memoryIDs, nil
}

// getBatchEmbeddings efficiently gets embeddings for multiple texts
func (b *BatchMemoryProcessor) getBatchEmbeddings(ctx context.Context, items []BatchMemoryItem) ([][]float32, error) {
	if b.memory.embeddingProvider == nil {
		return nil, errors.New("no embedding provider available")
	}

	// Extract all content strings
	contents := make([]string, len(items))
	for i, item := range items {
		contents[i] = item.Content
	}

	// If the provider supports batch operations, use it
	if batchProvider, ok := b.memory.embeddingProvider.(BatchEmbeddingProvider); ok {
		return batchProvider.GetBatchEmbeddings(ctx, contents)
	}

	// Otherwise, fall back to parallel individual operations
	embeddings := make([][]float32, len(contents))
	var wg sync.WaitGroup
	var mu sync.Mutex
	errChan := make(chan error, len(contents))

	for i, text := range contents {
		if text == "" {
			continue
		}

		wg.Add(1)
		go func(idx int, content string) {
			defer wg.Done()

			embedding, err := b.memory.embeddingProvider.GetEmbedding(ctx, content)
			if err != nil {
				errChan <- fmt.Errorf("failed to get embedding for text %d: %w", idx, err)
				return
			}

			mu.Lock()
			embeddings[idx] = embedding
			mu.Unlock()
		}(i, text)
	}

	wg.Wait()
	close(errChan)

	// Check if any errors occurred
	select {
	case err := <-errChan:
		return embeddings, err
	default:
		// No errors
	}

	return embeddings, nil
}

// BatchRetrieveMemories retrieves multiple sets of memories for different queries
func (b *BatchMemoryProcessor) BatchRetrieveMemories(
	ctx context.Context,
	queries []string,
	agentID string,
	limitPerQuery int,
	threshold float32,
) ([][]EnhancedMemoryEntry, error) {
	if len(queries) == 0 {
		return nil, nil
	}

	output.Action("memory", "batch_retrieve",
		fmt.Sprintf("Batch retrieving memories for %d queries", len(queries)))

	// First get embeddings for all queries
	queryEmbeddings := make([][]float32, len(queries))
	for i, query := range queries {
		embedding, err := b.memory.embeddingProvider.GetEmbedding(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("failed to get embedding for query %d: %w", i, err)
		}
		queryEmbeddings[i] = embedding
	}

	// If vector store supports batch search, use it
	if batchStore, ok := b.memory.vectorStore.(BatchVectorStore); ok {
		// Prepare filters for each query
		filtersList := make([]map[string]interface{}, len(queries))
		for i := range queries {
			if agentID != "" {
				filtersList[i] = map[string]interface{}{
					"agent_id": agentID,
				}
			} else {
				filtersList[i] = map[string]interface{}{}
			}
		}

		// Perform batch search
		searchResults, err := batchStore.BatchSearch(ctx, queryEmbeddings, limitPerQuery, filtersList)
		if err != nil {
			return nil, fmt.Errorf("batch search failed: %w", err)
		}

		// Convert search results to memory entries
		results := make([][]EnhancedMemoryEntry, len(searchResults))
		for i, queryResults := range searchResults {
			memories := make([]EnhancedMemoryEntry, 0, len(queryResults))
			for _, result := range queryResults {
				if result.Score < threshold {
					continue
				}

				// Retrieve the full memory entry
				b.memory.mutex.RLock()
				memEntry, exists := b.memory.memoryData[result.ID]
				b.memory.mutex.RUnlock()

				if exists {
					// Update access statistics
					b.memory.mutex.Lock()
					memEntry.AccessCount++
					memEntry.LastAccess = time.Now()
					if memEntry.UsageStats != nil {
						memEntry.UsageStats.RecordAccess()
						memEntry.UsageStats.RecordQueryMatch()
					}
					b.memory.mutex.Unlock()

					entry := *memEntry // Make a copy
					entry.Score = result.Score
					memories = append(memories, entry)
				}
			}
			results[i] = memories
		}

		return results, nil
	}

	// Otherwise, fall back to individual searches
	results := make([][]EnhancedMemoryEntry, len(queries))
	var wg sync.WaitGroup
	var mu sync.Mutex
	errChan := make(chan error, len(queries))

	for i, query := range queries {
		wg.Add(1)
		go func(idx int, q string) {
			defer wg.Done()

			// Use the regular retrieval method
			memories, err := b.memory.RetrieveMemoriesByVector(ctx, q, agentID, limitPerQuery, threshold)
			if err != nil {
				errChan <- fmt.Errorf("query %d failed: %w", idx, err)
				return
			}

			mu.Lock()
			results[idx] = memories
			mu.Unlock()
		}(i, query)
	}

	wg.Wait()
	close(errChan)

	// Check if any errors occurred
	select {
	case err := <-errChan:
		return results, err
	default:
		// No errors
	}

	output.Result(fmt.Sprintf("Retrieved memories for %d queries", len(queries)))
	return results, nil
}
