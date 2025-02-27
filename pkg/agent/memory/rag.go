/*
Package memory provides memory-related functionality for the agent system.
This file contains implementations for RAG (Retrieval Augmented Generation)
to enhance context with relevant memories for better responses.
*/
package memory

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/output"
)

// RAGOptions defines configuration options for RAG operations
type RAGOptions struct {
	// MaxContextSize is the maximum number of tokens allowed in context
	MaxContextSize int
	// MaxMemories is the maximum number of memories to include
	MaxMemories int
	// RelevanceThreshold is the minimum relevance score (0-1) for inclusion
	RelevanceThreshold float32
	// IncludeMetadata indicates whether to include memory metadata
	IncludeMetadata bool
	// ChunkSize is the size of chunks to split documents into
	ChunkSize int
	// ChunkOverlap is the overlap between chunks
	ChunkOverlap int
	// RetrievalStrategy determines how memories are selected (mmr, topk, hybrid)
	RetrievalStrategy string
	// MMRLambda is the diversity parameter for MMR (0=max diversity, 1=max relevance)
	MMRLambda float32
	// IncludeImportanceScores includes memory importance scores in the context
	IncludeImportanceScores bool
	// AnnotateRelevance adds relevance information to each memory
	AnnotateRelevance bool
	// EnableQueryRewriting enables automatic query variation generation
	EnableQueryRewriting bool
}

// DefaultRAGOptions returns default RAG options
func DefaultRAGOptions() *RAGOptions {
	return &RAGOptions{
		MaxContextSize:          4000,
		MaxMemories:             10,
		RelevanceThreshold:      0.6,
		IncludeMetadata:         true,
		ChunkSize:               1000,
		ChunkOverlap:            200,
		RetrievalStrategy:       "hybrid",
		MMRLambda:               0.7,
		IncludeImportanceScores: true,
		AnnotateRelevance:       true,
		EnableQueryRewriting:    true,
	}
}

// RAGProcessor handles RAG operations
type RAGProcessor struct {
	// memory is the reference to the unified memory system
	memory *UnifiedMemory
	// calculator is used to calculate memory importance
	calculator *MemoryImportanceCalculator
}

// NewRAGProcessor creates a new RAG processor
func NewRAGProcessor(memory *UnifiedMemory) *RAGProcessor {
	return &RAGProcessor{
		memory:     memory,
		calculator: NewMemoryImportanceCalculator(),
	}
}

// PrepareRAGContext prepares an enhanced context for LLM response generation
func (r *RAGProcessor) PrepareRAGContext(ctx context.Context, agentID string, query string, options *RAGOptions) (string, error) {
	if options == nil {
		options = DefaultRAGOptions()
	}

	output.Action("memory", "rag_context",
		fmt.Sprintf("Preparing RAG context for query: %s", output.Summarize(query, 40)))

	// Step 1: Generate query variations if enabled
	queries := []string{query}
	if options.EnableQueryRewriting {
		variations, err := r.generateQueryVariations(ctx, query)
		if err == nil && len(variations) > 0 {
			queries = append(queries, variations...)
		}
	}

	// Step 2: Retrieve memories for all query variations
	allMemories := make([]EnhancedMemoryEntry, 0)
	seenIDs := make(map[string]bool)

	for _, q := range queries {
		memories, err := r.memory.RetrieveMemoriesByVector(ctx, q, agentID, options.MaxMemories, options.RelevanceThreshold)
		if err != nil {
			continue
		}

		// Add to collection, avoiding duplicates
		for _, memory := range memories {
			if !seenIDs[memory.ID] {
				seenIDs[memory.ID] = true

				// Calculate and set importance score
				if options.IncludeImportanceScores {
					memory.ImportanceScore = r.calculator.CalculateImportance(ctx, &memory, r.memory)
				}

				allMemories = append(allMemories, memory)
			}
		}
	}

	// Record access for each retrieved memory
	r.recordMemoryAccess(ctx, allMemories)

	// Step 3: Apply retrieval strategy to select final memories
	var selectedMemories []EnhancedMemoryEntry

	if len(allMemories) > options.MaxMemories {
		switch options.RetrievalStrategy {
		case "mmr":
			selectedMemories = r.applyMMR(allMemories, options.MaxMemories, options.MMRLambda)
		case "hybrid":
			// First half from top relevance, second half from MMR
			halfCount := options.MaxMemories / 2
			r.sortByRelevance(allMemories)
			topRelevant := allMemories[:halfCount]
			rest := allMemories[halfCount:]
			diverseSet := r.applyMMR(rest, options.MaxMemories-halfCount, options.MMRLambda)
			selectedMemories = append(topRelevant, diverseSet...)
		default: // "topk"
			r.sortByRelevance(allMemories)
			selectedMemories = allMemories[:options.MaxMemories]
		}
	} else {
		selectedMemories = allMemories
	}

	// Step 4: Format selected memories into context
	result := r.formatRAGContext(query, selectedMemories, options)

	output.Result(fmt.Sprintf("Prepared RAG context with %d memories", len(selectedMemories)))
	return result, nil
}

// generateQueryVariations creates different versions of a query for better recall
func (r *RAGProcessor) generateQueryVariations(ctx context.Context, query string) ([]string, error) {
	// If the memory system has a query generation method, use it
	if r.memory != nil {
		queries, err := r.memory.GenerateMemoryQueries(ctx, query)
		if err == nil && len(queries) > 0 {
			return queries, nil
		}
	}

	// Fallback to simple variations if no advanced method is available
	variations := []string{
		fmt.Sprintf("Information about %s", query),
		fmt.Sprintf("Tell me about %s", query),
		fmt.Sprintf("What do you know about %s", query),
	}

	return variations, nil
}

// recordMemoryAccess records that these memories were accessed
func (r *RAGProcessor) recordMemoryAccess(ctx context.Context, memories []EnhancedMemoryEntry) {
	r.memory.mutex.Lock()
	defer r.memory.mutex.Unlock()

	for _, memory := range memories {
		// Update in-memory data
		if storedMemory, exists := r.memory.memoryData[memory.ID]; exists {
			// Ensure UsageStats exists
			if storedMemory.UsageStats == nil {
				storedMemory.UsageStats = NewMemoryUsageStats()
			}

			// Record access and query match
			storedMemory.UsageStats.RecordAccess()
			storedMemory.UsageStats.RecordQueryMatch()

			// Update regular access stats too for backward compatibility
			storedMemory.AccessCount++
			storedMemory.LastAccess = time.Now()
		}
	}
}

// sortByRelevance sorts memories by relevance score (descending)
func (r *RAGProcessor) sortByRelevance(memories []EnhancedMemoryEntry) {
	sort.Slice(memories, func(i, j int) bool {
		return memories[i].Score > memories[j].Score
	})
}

// applyMMR applies Maximum Marginal Relevance algorithm for diversity
func (r *RAGProcessor) applyMMR(memories []EnhancedMemoryEntry, limit int, lambda float32) []EnhancedMemoryEntry {
	if len(memories) <= limit {
		return memories
	}

	// Sort initially by relevance
	r.sortByRelevance(memories)

	// Take the most relevant item first
	selected := []EnhancedMemoryEntry{memories[0]}
	candidates := memories[1:]

	// Select the rest using MMR
	for len(selected) < limit && len(candidates) > 0 {
		maxScore := float32(-1.0)
		maxIdx := 0

		for i, candidate := range candidates {
			// Relevance score (already normalized)
			relevanceScore := candidate.Score

			// Diversity score (maximum similarity to any selected item)
			var maxSimilarity float32 = 0
			for _, item := range selected {
				similarity := cosineSimilarity(candidate.Embedding, item.Embedding)
				if similarity > maxSimilarity {
					maxSimilarity = similarity
				}
			}

			// MMR score = λ * relevance - (1 - λ) * maxSimilarity
			mmrScore := lambda*relevanceScore - (1-lambda)*maxSimilarity

			if mmrScore > maxScore {
				maxScore = mmrScore
				maxIdx = i
			}
		}

		// Add the item with the highest MMR score
		selected = append(selected, candidates[maxIdx])

		// Remove from candidates
		candidates = append(candidates[:maxIdx], candidates[maxIdx+1:]...)
	}

	return selected
}

// formatRAGContext formats memories into a context string
func (r *RAGProcessor) formatRAGContext(query string, memories []EnhancedMemoryEntry, options *RAGOptions) string {
	var sb strings.Builder

	sb.WriteString("## Relevant Information\n\n")

	for i, memory := range memories {
		// Write memory header with metadata
		sb.WriteString(fmt.Sprintf("### Memory %d\n", i+1))

		if options.AnnotateRelevance {
			sb.WriteString(fmt.Sprintf("Relevance: %.2f", memory.Score))
			if options.IncludeImportanceScores {
				sb.WriteString(fmt.Sprintf(" | Importance: %.2f", memory.ImportanceScore))
			}
			sb.WriteString("\n")
		}

		// Include metadata if requested
		if options.IncludeMetadata {
			sb.WriteString(fmt.Sprintf("Source: %s | Created: %s",
				memory.Source, memory.CreatedAt.Format(time.RFC822)))

			// Add any custom metadata
			if len(memory.Metadata) > 0 {
				sb.WriteString(" | ")
				first := true
				for k, v := range memory.Metadata {
					if !first {
						sb.WriteString(", ")
					}
					sb.WriteString(fmt.Sprintf("%s: %s", k, v))
					first = false
				}
			}
			sb.WriteString("\n")
		}

		// Add the memory content
		sb.WriteString("\n")
		sb.WriteString(memory.Content)
		sb.WriteString("\n\n")

		// Add separator between memories
		sb.WriteString("---\n\n")
	}

	// Add the original query at the end
	sb.WriteString(fmt.Sprintf("## Original Query\n%s\n", query))

	return sb.String()
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
