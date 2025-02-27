/*
Package memory provides memory-related functionality for the agent system.
This file contains the implementation of memory forgetting mechanisms.
*/
package memory

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/output"
)

// ForgetOptions defines options for memory forgetting
type ForgetOptions struct {
	// ImportanceThreshold is the minimum importance score to retain
	ImportanceThreshold float32
	// AgentID is the agent ID to filter memories by (empty for all agents)
	AgentID string
	// SourceFilter is a source to filter memories by (empty for all sources)
	SourceFilter string
	// IncludeGlobal indicates whether to include global memories
	IncludeGlobal bool
	// DryRun performs a dry run without actually deleting
	DryRun bool
}

// DefaultForgetOptions returns default forgetting options
func DefaultForgetOptions() *ForgetOptions {
	return &ForgetOptions{
		ImportanceThreshold: 0.3,
		AgentID:             "", // No agent filter
		SourceFilter:        "", // No source filter
		IncludeGlobal:       false,
		DryRun:              false,
	}
}

// MemoryForgetter provides mechanisms for forgetting memories
type MemoryForgetter struct {
	// memory is the reference to the unified memory system
	memory *UnifiedMemory
	// calculator is used to calculate memory importance
	calculator *MemoryImportanceCalculator
}

// NewMemoryForgetter creates a new memory forgetter
func NewMemoryForgetter(memory *UnifiedMemory) *MemoryForgetter {
	return &MemoryForgetter{
		memory:     memory,
		calculator: NewMemoryImportanceCalculator(),
	}
}

// ForgetByThreshold forgets memories with importance below a threshold
func (f *MemoryForgetter) ForgetByThreshold(ctx context.Context, options *ForgetOptions) (int, error) {
	if options == nil {
		options = DefaultForgetOptions()
	}

	output.Action("memory", "forget_by_threshold",
		fmt.Sprintf("Forgetting memories below threshold: %.2f", options.ImportanceThreshold))

	// Step 1: Find memories below the threshold
	var memoriesToForget []string

	f.memory.mutex.RLock()
	for id, memory := range f.memory.memoryData {
		// Apply filters
		if !f.shouldIncludeMemory(memory, options) {
			continue
		}

		// Calculate importance score
		importanceScore := f.calculator.CalculateImportance(ctx, memory, f.memory)

		// Update the score in the memory (helpful for debugging and analysis)
		if !options.DryRun {
			memory.ImportanceScore = importanceScore
		}

		// Check if below threshold
		if importanceScore < options.ImportanceThreshold {
			memoriesToForget = append(memoriesToForget, id)
		}
	}
	f.memory.mutex.RUnlock()

	// Step 2: Forget each identified memory
	count := 0
	if !options.DryRun {
		for _, id := range memoriesToForget {
			if err := f.memory.ForgetMemory(ctx, id); err != nil {
				output.Error(fmt.Sprintf("Failed to forget memory %s", id), err)
				continue
			}
			count++
		}
	} else {
		count = len(memoriesToForget)
		output.Debug(fmt.Sprintf("Dry run identified %d memories below threshold %.2f",
			count, options.ImportanceThreshold))
	}

	output.Result(fmt.Sprintf("Forgot %d memories below threshold %.2f", count, options.ImportanceThreshold))
	return count, nil
}

// ForgetByAge forgets memories older than the specified age
func (f *MemoryForgetter) ForgetByAge(ctx context.Context, age time.Duration, options *ForgetOptions) (int, error) {
	if options == nil {
		options = DefaultForgetOptions()
	}

	output.Action("memory", "forget_by_age", fmt.Sprintf("Forgetting memories older than %v", age))

	cutoffTime := time.Now().Add(-age)
	var memoriesToForget []string

	f.memory.mutex.RLock()
	for id, memory := range f.memory.memoryData {
		// Apply filters
		if !f.shouldIncludeMemory(memory, options) {
			continue
		}

		// Check age criteria - memory is old AND hasn't been accessed recently
		if memory.CreatedAt.Before(cutoffTime) {
			// If it has recent access, check importance score as a secondary filter
			if !memory.LastAccess.Before(cutoffTime) {
				// Recently accessed - only forget if importance is very low
				importanceScore := f.calculator.CalculateImportance(ctx, memory, f.memory)
				if importanceScore >= options.ImportanceThreshold {
					continue
				}
			}
			memoriesToForget = append(memoriesToForget, id)
		}
	}
	f.memory.mutex.RUnlock()

	// Forget each identified memory
	count := 0
	if !options.DryRun {
		for _, id := range memoriesToForget {
			if err := f.memory.ForgetMemory(ctx, id); err != nil {
				output.Error(fmt.Sprintf("Failed to forget memory %s", id), err)
				continue
			}
			count++
		}
	} else {
		count = len(memoriesToForget)
		output.Debug(fmt.Sprintf("Dry run identified %d memories older than %v", count, age))
	}

	output.Result(fmt.Sprintf("Forgot %d memories older than %v", count, age))
	return count, nil
}

// ForgetByQuery forgets memories that match a specific query
func (f *MemoryForgetter) ForgetByQuery(ctx context.Context, query string, options *ForgetOptions) (int, error) {
	if options == nil {
		options = DefaultForgetOptions()
	}

	output.Action("memory", "forget_by_query",
		fmt.Sprintf("Forgetting memories matching query: %s", output.Summarize(query, 40)))

	// First, find memories that match the query
	memories, err := f.memory.RetrieveMemoriesByVector(ctx, query, options.AgentID, 100, 0.5)
	if err != nil {
		return 0, fmt.Errorf("failed to retrieve memories by query: %w", err)
	}

	// Filter memories based on options
	var memoriesToForget []string
	for _, memory := range memories {
		if !f.shouldIncludeMemory(&memory, options) {
			continue
		}

		memoriesToForget = append(memoriesToForget, memory.ID)
	}

	// Forget each identified memory
	count := 0
	if !options.DryRun {
		for _, id := range memoriesToForget {
			if err := f.memory.ForgetMemory(ctx, id); err != nil {
				output.Error(fmt.Sprintf("Failed to forget memory %s", id), err)
				continue
			}
			count++
		}
	} else {
		count = len(memoriesToForget)
		output.Debug(fmt.Sprintf("Dry run identified %d memories matching query", count))
	}

	output.Result(fmt.Sprintf("Forgot %d memories matching query", count))
	return count, nil
}

// ArchiveMemory moves a memory to long-term cold storage
func (f *MemoryForgetter) ArchiveMemory(ctx context.Context, memoryID string) error {
	output.Action("memory", "archive", fmt.Sprintf("Archiving memory: %s", memoryID))

	// Retrieve the memory to archive
	f.memory.mutex.RLock()
	memory, exists := f.memory.memoryData[memoryID]
	f.memory.mutex.RUnlock()

	if !exists {
		return fmt.Errorf("memory not found: %s", memoryID)
	}

	// Store the memory in archive storage
	// NOTE: This is a simplified implementation. In a production system,
	// you would have a separate archive storage system or table.

	// For now, we'll just mark it as archived in its metadata
	f.memory.mutex.Lock()
	if memory.Metadata == nil {
		memory.Metadata = make(map[string]string)
	}
	memory.Metadata["archived"] = "true"
	memory.Metadata["archive_date"] = time.Now().Format(time.RFC3339)
	f.memory.mutex.Unlock()

	// Remove from active vector and graph stores
	if f.memory.options.EnableVectorStore && f.memory.vectorStore != nil {
		if err := f.memory.vectorStore.Delete(ctx, memoryID); err != nil {
			return fmt.Errorf("failed to remove from vector store: %w", err)
		}
	}

	output.Result(fmt.Sprintf("Successfully archived memory: %s", memoryID))
	return nil
}

// shouldIncludeMemory checks if a memory should be included based on filtering options
func (f *MemoryForgetter) shouldIncludeMemory(memory *EnhancedMemoryEntry, options *ForgetOptions) bool {
	// Filter by agent ID if specified
	if options.AgentID != "" && memory.AgentID != options.AgentID {
		return false
	}

	// Filter by source if specified
	if options.SourceFilter != "" && memory.Source != options.SourceFilter {
		return false
	}

	// Exclude global memories if specified
	if !options.IncludeGlobal && memory.Type == MemoryTypeGlobal {
		return false
	}

	return true
}
