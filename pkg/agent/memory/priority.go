/*
Package memory provides memory-related functionality for the agent system.
This file contains the implementation of memory prioritization mechanisms.
*/
package memory

import (
	"context"
	"math"
	"time"
)

// MemoryUsageStats tracks usage patterns for a memory entry
type MemoryUsageStats struct {
	// AccessCount tracks how many times this memory has been accessed
	AccessCount int
	// LastAccess records when this memory was last accessed
	LastAccess time.Time
	// AccessHistory stores recent access timestamps (circular buffer)
	AccessHistory []time.Time
	// QueryMatches counts how many times this matched a query
	QueryMatches int
	// QueryHelpfulness tracks feedback score on how helpful this memory was
	QueryHelpfulness float32
	// MaxHistorySize defines how many recent accesses to track
	MaxHistorySize int
}

// NewMemoryUsageStats creates a new MemoryUsageStats with default values
func NewMemoryUsageStats() *MemoryUsageStats {
	return &MemoryUsageStats{
		AccessCount:      0,
		LastAccess:       time.Now(),
		AccessHistory:    make([]time.Time, 0, 10),
		QueryMatches:     0,
		QueryHelpfulness: 0.5,
		MaxHistorySize:   10,
	}
}

// RecordAccess records a memory access event
func (m *MemoryUsageStats) RecordAccess() {
	m.AccessCount++
	m.LastAccess = time.Now()

	// Add to access history with circular buffer behavior
	if len(m.AccessHistory) >= m.MaxHistorySize {
		// Remove oldest entry
		m.AccessHistory = m.AccessHistory[1:]
	}
	m.AccessHistory = append(m.AccessHistory, m.LastAccess)
}

// RecordQueryMatch records that this memory matched a query
func (m *MemoryUsageStats) RecordQueryMatch() {
	m.QueryMatches++
}

// UpdateHelpfulness updates the helpfulness score based on feedback
func (m *MemoryUsageStats) UpdateHelpfulness(score float32) {
	// Weighted average with new score (70% old, 30% new)
	m.QueryHelpfulness = (0.7 * m.QueryHelpfulness) + (0.3 * score)
}

// MemoryImportanceCalculator calculates importance scores for memories
type MemoryImportanceCalculator struct {
	// Weights for different importance factors
	RecencyWeight     float32
	UsageWeight       float32
	CentralityWeight  float32
	UniquenessWeight  float32
	HelpfulnessWeight float32
}

// NewMemoryImportanceCalculator creates a new calculator with default weights
func NewMemoryImportanceCalculator() *MemoryImportanceCalculator {
	return &MemoryImportanceCalculator{
		RecencyWeight:     0.3,
		UsageWeight:       0.25,
		CentralityWeight:  0.15,
		UniquenessWeight:  0.15,
		HelpfulnessWeight: 0.15,
	}
}

// CalculateImportance calculates the overall importance score for a memory
func (c *MemoryImportanceCalculator) CalculateImportance(ctx context.Context, memory *EnhancedMemoryEntry, um *UnifiedMemory) float32 {
	// Start with a base score
	score := float32(0.5)

	// Factor 1: Recency - Newer memories start with higher importance
	recencyScore := c.calculateRecencyScore(memory)

	// Factor 2: Usage - More frequently accessed memories are more important
	usageScore := c.calculateUsageScore(memory)

	// Factor 3: Network centrality - If it has many relationships in the graph
	centralityScore := float32(0.0)
	if um.graphStore != nil {
		centralityScore = c.calculateCentralityScore(ctx, memory.ID, um)
	}

	// Factor 4: Content uniqueness - More unique content is more valuable
	uniquenessScore := c.calculateUniquenessScore(ctx, memory, um)

	// Factor 5: Helpfulness in previous queries
	helpfulnessScore := float32(0.0)
	if memory.UsageStats != nil {
		helpfulnessScore = memory.UsageStats.QueryHelpfulness
	}

	// Combine all factors with appropriate weights
	score = (c.RecencyWeight * recencyScore) +
		(c.UsageWeight * usageScore) +
		(c.CentralityWeight * centralityScore) +
		(c.UniquenessWeight * uniquenessScore) +
		(c.HelpfulnessWeight * helpfulnessScore)

	return score
}

// calculateRecencyScore determines how recent a memory is
func (c *MemoryImportanceCalculator) calculateRecencyScore(memory *EnhancedMemoryEntry) float32 {
	// Calculate age in days
	ageInDays := float32(time.Since(memory.CreatedAt).Hours() / 24)

	// Apply half-life decay function (half-life of 30 days)
	// This means memories 30 days old have half the importance of new ones
	recencyScore := float32(1.0) / (1.0 + (ageInDays / 30.0))

	return recencyScore
}

// calculateUsageScore determines importance based on access patterns
func (c *MemoryImportanceCalculator) calculateUsageScore(memory *EnhancedMemoryEntry) float32 {
	if memory.UsageStats == nil || memory.UsageStats.AccessCount == 0 {
		return 0.0
	}

	// Use log scale to prevent very frequent items from dominating
	// Log(1+x) ensures the value is positive and grows more slowly for large x
	usageScore := float32(0.3) * float32(math.Log1p(float64(memory.UsageStats.AccessCount)))

	// Recent usage is more important than old usage
	daysSinceLastAccess := float32(time.Since(memory.UsageStats.LastAccess).Hours() / 24)
	usageRecencyFactor := float32(1.0) / (1.0 + (daysSinceLastAccess / 7.0))
	usageScore *= usageRecencyFactor

	// Cap the maximum score at 1.0
	if usageScore > 1.0 {
		usageScore = 1.0
	}

	return usageScore
}

// calculateCentralityScore determines how central a memory is in the knowledge graph
func (c *MemoryImportanceCalculator) calculateCentralityScore(ctx context.Context, memoryID string, um *UnifiedMemory) float32 {
	if um.graphStore == nil {
		return 0.0
	}

	// Query the graph store for the number of relationships this memory has
	query := `
		MATCH (m:Memory {id: $id})-[r]-(other)
		RETURN count(r) as relationCount
	`
	params := map[string]interface{}{
		"id": memoryID,
	}

	results, err := um.graphStore.Query(ctx, query, params)
	if err != nil || len(results) == 0 {
		return 0.0
	}

	// Get the count from the results
	if count, ok := results[0]["relationCount"].(int64); ok {
		// Normalize: 0 relationships = 0.0, 10+ relationships = 1.0
		return float32(math.Min(1.0, float64(count)/10.0))
	}

	return 0.0
}

// calculateUniquenessScore determines how unique this memory's content is
func (c *MemoryImportanceCalculator) calculateUniquenessScore(ctx context.Context, memory *EnhancedMemoryEntry, um *UnifiedMemory) float32 {
	if um.vectorStore == nil || len(memory.Embedding) == 0 {
		return 0.5 // Default middle value when we can't calculate
	}

	// Find similar memories
	similar, err := um.vectorStore.Search(ctx, memory.Embedding, 5, map[string]interface{}{
		"id_not_equal": memory.ID, // Exclude the memory itself
	})

	if err != nil || len(similar) == 0 {
		return 1.0 // Assume unique if error or no similar items found
	}

	// Calculate average similarity of the most similar items
	var totalSimilarity float32 = 0.0
	for _, item := range similar {
		totalSimilarity += item.Score
	}

	avgSimilarity := totalSimilarity / float32(len(similar))

	// Convert to uniqueness (1.0 - similarity)
	return 1.0 - avgSimilarity
}

// RelevanceCache caches relevance scores for common queries
type RelevanceCache struct {
	// QueryScores maps query signatures to relevance scores
	QueryScores map[string]float32
	// MaxSize is the maximum number of queries to cache
	MaxSize int
}

// NewRelevanceCache creates a new relevance cache with default settings
func NewRelevanceCache() *RelevanceCache {
	return &RelevanceCache{
		QueryScores: make(map[string]float32),
		MaxSize:     50,
	}
}

// StoreScore stores a relevance score for a query
func (rc *RelevanceCache) StoreScore(query string, score float32) {
	// Simple cleanup if we exceed max size
	if len(rc.QueryScores) >= rc.MaxSize {
		// Just clear everything for simplicity
		// In a production system, we'd use LRU or more sophisticated eviction
		rc.QueryScores = make(map[string]float32)
	}

	rc.QueryScores[query] = score
}

// GetScore retrieves a cached relevance score for a query
func (rc *RelevanceCache) GetScore(query string) (float32, bool) {
	score, exists := rc.QueryScores[query]
	return score, exists
}
