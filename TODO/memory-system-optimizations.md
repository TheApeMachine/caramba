# Memory System Optimizations

## 1.1 Memory Prioritization System

**Files to Modify:**
- `pkg/agent/memory/unified_memory.go`
- `pkg/agent/memory/schema.go`

**Implementation Details:**

1. Extend `EnhancedMemoryEntry` in `memory.go`:

```go
type EnhancedMemoryEntry struct {
    // Existing fields...
    
    // New fields for prioritization
    ImportanceScore    float32               // Algorithmically determined importance
    RelevanceCache     map[string]float32    // Cache of relevance scores to common queries
    UsageStats         MemoryUsageStats      // Enhanced usage statistics
}

type MemoryUsageStats struct {
    AccessCount        int
    LastAccess         time.Time
    AccessHistory      []time.Time  // Limited circular buffer of last N access times
    QueryMatches       int          // How many times this matched a query
    QueryHelpfulness   float32      // Feedback score on how helpful this memory was
}
```

2. Add importance scoring algorithm in `unified_memory.go`:

```go
// CalculateImportanceScore determines a memory's importance based on multiple factors
func (um *UnifiedMemory) CalculateImportanceScore(ctx context.Context, memory *EnhancedMemoryEntry) float32 {
    // Start with a base score
    score := float32(0.5)
    
    // Factor 1: Recency - Newer memories start with higher importance
    ageInDays := float32(time.Since(memory.CreatedAt).Hours() / 24)
    recencyScore := float32(1.0) / (1.0 + (ageInDays / 30.0)) // Half-life of 30 days
    
    // Factor 2: Usage - More frequently accessed memories are more important
    usageScore := float32(0.0)
    if memory.UsageStats.AccessCount > 0 {
        // Log scale to prevent very frequent items from dominating
        usageScore = float32(0.3) * float32(math.Log1p(float64(memory.UsageStats.AccessCount)))
        
        // Recent usage is more important than old usage
        daysSinceLastAccess := float32(time.Since(memory.UsageStats.LastAccess).Hours() / 24)
        usageRecencyFactor := float32(1.0) / (1.0 + (daysSinceLastAccess / 7.0))
        usageScore *= usageRecencyFactor
    }
    
    // Factor 3: Network centrality - If it has many relationships in the graph
    centralityScore := float32(0.0)
    if um.graphStore != nil {
        centralityScore = um.calculateCentralityScore(ctx, memory.ID)
    }
    
    // Factor 4: Content uniqueness - More unique content is more valuable
    uniquenessScore := um.calculateUniquenessScore(ctx, memory)
    
    // Combine all factors with appropriate weights
    score = (0.3 * recencyScore) + (0.3 * usageScore) + (0.2 * centralityScore) + (0.2 * uniquenessScore)
    
    return score
}

// calculateCentralityScore determines how central a memory is in the knowledge graph
func (um *UnifiedMemory) calculateCentralityScore(ctx context.Context, memoryID string) float32 {
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
func (um *UnifiedMemory) calculateUniquenessScore(ctx context.Context, memory *EnhancedMemoryEntry) float32 {
    if um.vectorStore == nil || len(memory.Embedding) == 0 {
        return 0.5 // Default middle value when we can't calculate
    }
    
    // Find similar memories
    similar, err := um.vectorStore.Search(ctx, memory.Embedding, 5, nil)
    if err != nil || len(similar) <= 1 {
        return 1.0 // Assume unique if error or no similar items found
    }
    
    // Skip the first result (which is the memory itself)
    var totalSimilarity float32 = 0.0
    for i := 1; i < len(similar); i++ {
        totalSimilarity += similar[i].Score
    }
    
    // Average similarity of the most similar items
    avgSimilarity := totalSimilarity / float32(len(similar)-1)
    
    // Convert to uniqueness (1.0 - similarity)
    return 1.0 - avgSimilarity
}
```

3. Add memory reranking function to optimize retrieval:

```go
// RetrieveAndRankMemories retrieves memories with intelligent reranking
func (um *UnifiedMemory) RetrieveAndRankMemories(
    ctx context.Context,
    query string,
    agentID string,
    limit int,
    options *MemoryRetrievalOptions,
) ([]EnhancedMemoryEntry, error) {
    if options == nil {
        options = DefaultMemoryRetrievalOptions()
    }
    
    // First get base results from vector search
    baseMemories, err := um.RetrieveMemoriesByVector(ctx, query, agentID, limit*2, 0.5)
    if err != nil {
        return nil, err
    }
    
    // Apply reranking with the composite score
    for i := range baseMemories {
        // Start with the vector similarity score
        vectorScore := baseMemories[i].Score
        
        // Get or calculate importance score
        importanceScore := um.CalculateImportanceScore(ctx, &baseMemories[i])
        
        // Calculate final score as weighted combination
        baseMemories[i].Score = 
            (options.VectorSimilarityWeight * vectorScore) + 
            (options.ImportanceWeight * importanceScore)
            
        // Update memory access stats
        um.updateMemoryAccessStats(ctx, baseMemories[i].ID)
    }
    
    // Sort by new score
    sort.Slice(baseMemories, func(i, j int) bool {
        return baseMemories[i].Score > baseMemories[j].Score
    })
    
    // Return top results after reranking
    if len(baseMemories) > limit {
        baseMemories = baseMemories[:limit]
    }
    
    return baseMemories, nil
}

// Define options for memory retrieval
type MemoryRetrievalOptions struct {
    VectorSimilarityWeight float32
    ImportanceWeight      float32
    RecencyBoost          bool
    UsageBoost            bool
    DiversityReranking    bool
}

// DefaultMemoryRetrievalOptions creates default options
func DefaultMemoryRetrievalOptions() *MemoryRetrievalOptions {
    return &MemoryRetrievalOptions{
        VectorSimilarityWeight: 0.7,
        ImportanceWeight:      0.3,
        RecencyBoost:          true,
        UsageBoost:            true,
        DiversityReranking:    false,
    }
}
```

## 1.2 Forgetting Mechanisms

**Files to Modify:**
- `pkg/agent/memory/unified_memory.go`
- `pkg/agent/core/interfaces.go` (to add forgetting interface)

**Implementation Details:**

1. Add memory forgetting interface in `interfaces.go`:

```go
// MemoryManager defines enhanced memory management operations
type MemoryManager interface {
    Memory
    
    // ForgetByThreshold forgets memories below a certain importance threshold
    ForgetByThreshold(ctx context.Context, threshold float32) (int, error)
    
    // ForgetByAge forgets memories older than the specified duration
    ForgetByAge(ctx context.Context, age time.Duration) (int, error)
    
    // ForgetByQuery forgets memories that match a specific query
    ForgetByQuery(ctx context.Context, query string) (int, error)
    
    // ArchiveMemory moves a memory to long-term cold storage
    ArchiveMemory(ctx context.Context, memoryID string) error
}
```

2. Implement forgetting mechanisms in `unified_memory.go`:

```go
// ForgetByThreshold forgets memories with importance below threshold
func (um *UnifiedMemory) ForgetByThreshold(ctx context.Context, threshold float32) (int, error) {
    output.Verbose(fmt.Sprintf("Forgetting memories below threshold: %.2f", threshold))
    
    // Step 1: Find memories below the threshold
    var memoriesToForget []string
    
    um.mutex.RLock()
    for id, memory := range um.memoryData {
        importanceScore := um.CalculateImportanceScore(ctx, memory)
        if importanceScore < threshold {
            memoriesToForget = append(memoriesToForget, id)
        }
    }
    um.mutex.RUnlock()
    
    // Step 2: Forget each identified memory
    count := 0
    for _, id := range memoriesToForget {
        if err := um.ForgetMemory(ctx, id); err != nil {
            output.Error(fmt.Sprintf("Failed to forget memory %s", id), err)
            continue
        }
        count++
    }
    
    output.Result(fmt.Sprintf("Forgot %d memories below threshold %.2f", count, threshold))
    return count, nil
}

// ForgetByAge forgets memories older than the specified age
func (um *UnifiedMemory) ForgetByAge(ctx context.Context, age time.Duration) (int, error) {
    output.Verbose(fmt.Sprintf("Forgetting memories older than %v", age))
    
    cutoffTime := time.Now().Add(-age)
    var memoriesToForget []string
    
    um.mutex.RLock()
    for id, memory := range um.memoryData {
        if memory.CreatedAt.Before(cutoffTime) && 
           memory.LastAccess.Before(cutoffTime) {
            memoriesToForget = append(memoriesToForget, id)
        }
    }
    um.mutex.RUnlock()
    
    // Forget each identified memory
    count := 0
    for _, id := range memoriesToForget {
        if err := um.ForgetMemory(ctx, id); err != nil {
            output.Error(fmt.Sprintf("Failed to forget memory %s", id), err)
            continue
        }
        count++
    }
    
    output.Result(fmt.Sprintf("Forgot %d memories older than %v", count, age))
    return count, nil
}

// ForgetMemory removes a specific memory from all storage systems
func (um *UnifiedMemory) ForgetMemory(ctx context.Context, memoryID string) error {
    um.mutex.Lock()
    defer um.mutex.Unlock()
    
    // Remove from in-memory cache
    delete(um.memoryData, memoryID)
    
    // Remove from vector store if enabled
    if um.options.EnableVectorStore && um.vectorStore != nil {
        if err := um.vectorStore.Delete(ctx, memoryID); err != nil {
            return fmt.Errorf("failed to remove from vector store: %w", err)
        }
    }
    
    // Remove from graph store if enabled
    if um.options.EnableGraphStore && um.graphStore != nil {
        if err := um.graphStore.DeleteNode(ctx, memoryID); err != nil {
            return fmt.Errorf("failed to remove from graph store: %w", err)
        }
    }
    
    // Remove from base store
    if err := um.baseStore.Delete(ctx, "memory:"+memoryID); err != nil {
        return fmt.Errorf("failed to remove from base store: %w", err)
    }
    
    output.Debug(fmt.Sprintf("Forgot memory: %s", memoryID))
    return nil
}
```

## 1.3 RAG (Retrieval Augmented Generation) Enhancement

**Files to Modify:**
- `pkg/agent/memory/context.go`
- `pkg/agent/memory/schema.go`

**New Files to Create:**
- `pkg/agent/memory/rag.go`

**Implementation Details:**

1. Create explicit RAG implementation in `rag.go`:

```go
package memory

import (
    "context"
    "fmt"
    "sort"
    "strings"
    "time"
    
    "github.com/theapemachine/caramba/pkg/output"
)

// RAGOptions defines configuration for RAG operations
type RAGOptions struct {
    MaxContextSize        int      // Maximum number of tokens allowed in context
    MaxMemories           int      // Maximum number of memories to include
    RelevanceThreshold    float32  // Minimum relevance score (0-1)
    IncludeMetadata       bool     // Whether to include memory metadata
    ChunkSize             int      // Size of chunks to split documents into
    ChunkOverlap          int      // Overlap between chunks
    RetrievalStrategy     string   // "mmr", "topk", or "hybrid"
    MMRLambda             float32  // Diversity parameter for MMR (0=max diversity, 1=max relevance)
}

// DefaultRAGOptions returns default RAG options
func DefaultRAGOptions() *RAGOptions {
    return &RAGOptions{
        MaxContextSize:     4000,
        MaxMemories:        10,
        RelevanceThreshold: 0.7,
        IncludeMetadata:    true,
        ChunkSize:          1000,
        ChunkOverlap:       200,
        RetrievalStrategy:  "hybrid",
        MMRLambda:          0.7,
    }
}

// PrepareRAGContext prepares context for RAG with advanced techniques
func (um *UnifiedMemory) PrepareRAGContext(ctx context.Context, agentID string, query string, options *RAGOptions) (string, error) {
    if options == nil {
        options = DefaultRAGOptions()
    }
    
    output.Action("memory", "rag_context", fmt.Sprintf("Preparing RAG context for query: %s", output.Summarize(query, 40)))
    
    // Generate multiple query variations for better recall
    queryVariations, err := um.GenerateQueryVariations(ctx, query)
    if err != nil {
        queryVariations = []string{query}
    }
    
    allMemories := make([]EnhancedMemoryEntry, 0)
    for _, queryVar := range queryVariations {
        // Retrieve memories for each query variation
        memories, err := um.RetrieveAndRankMemories(ctx, queryVar, agentID, options.MaxMemories, nil)
        if err != nil {
            continue
        }
        
        // Add to collection, tracking seen IDs to avoid duplicates
        seen := make(map[string]bool)
        for _, memory := range memories {
            if !seen[memory.ID] {
                seen[memory.ID] = true
                allMemories = append(allMemories, memory)
            }
        }
    }
    
    // If we retrieved too many memories, rerank and filter
    if len(allMemories) > options.MaxMemories {
        switch options.RetrievalStrategy {
        case "mmr":
            allMemories = um.applyMMR(allMemories, options.MaxMemories, options.MMRLambda)
        case "hybrid":
            // First half from top relevance, second half from MMR
            halfCount := options.MaxMemories / 2
            sortByRelevance(allMemories)
            topRelevant := allMemories[:halfCount]
            rest := allMemories[halfCount:]
            diverseSet := um.applyMMR(rest, options.MaxMemories-halfCount, options.MMRLambda)
            allMemories = append(topRelevant, diverseSet...)
        default: // "topk"
            sortByRelevance(allMemories)
            allMemories = allMemories[:options.MaxMemories]
        }
    }
    
    // Now format the memories for inclusion in context
    var sb strings.Builder
    sb.WriteString("Relevant information from memory:\n\n")
    
    for i, memory := range allMemories {
        sb.WriteString(fmt.Sprintf("Memory %d [Relevance: %.2f]:\n", i+1, memory.Score))
        sb.WriteString(memory.Content)
        sb.WriteString("\n\n")
        
        if options.IncludeMetadata {
            sb.WriteString(fmt.Sprintf("Source: %s | Created: %s\n", 
                memory.Source, memory.CreatedAt.Format(time.RFC822)))
        }
        
        sb.WriteString("---\n\n")
    }
    
    sb.WriteString(fmt.Sprintf("\nOriginal Query: %s\n", query))
    
    // Log summary of what we're returning
    output.Result(fmt.Sprintf("Prepared RAG context with %d memories", len(allMemories)))
    
    return sb.String(), nil
}

// GenerateQueryVariations creates multiple versions of the query for better recall
func (um *UnifiedMemory) GenerateQueryVariations(ctx context.Context, query string) ([]string, error) {
    // Start with the original query plus the ones we already had from GenerateMemoryQueries
    baseQueries, _ := um.GenerateMemoryQueries(ctx, query)
    
    // TODO: Use the LLM to generate more variations, synonyms, etc.
    
    return baseQueries, nil
}

// applyMMR applies Maximum Marginal Relevance algorithm for diversity
func (um *UnifiedMemory) applyMMR(memories []EnhancedMemoryEntry, limit int, lambda float32) []EnhancedMemoryEntry {
    if len(memories) <= limit {
        return memories
    }
    
    // Sort initially by relevance
    sortByRelevance(memories)
    
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
            
            // MMR score
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

// sortByRelevance sorts memories by relevance score (descending)
func sortByRelevance(memories []EnhancedMemoryEntry) {
    sort.Slice(memories, func(i, j int) bool {
        return memories[i].Score > memories[j].Score
    })
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
```

2. Modify the `PrepareContext` method in `context.go` to use the new RAG functionality:

```go
// PrepareContext enriches a prompt with relevant memories
func (um *UnifiedMemory) PrepareContext(ctx context.Context, agentID string, query string) (string, error) {
    output.Action("memory", "prepare_context", fmt.Sprintf("Enhancing context for agent %s", agentID))
    
    // Use the new RAG implementation with default options
    enhancedContext, err := um.PrepareRAGContext(ctx, agentID, query, nil)
    if err != nil {
        output.Error("Failed to prepare RAG context", err)
        return query, nil
    }
    
    output.Result("Enhanced context with relevant memories")
    return enhancedContext, nil
}
```

## 1.4 Batch Operations Optimization

**Files to Modify:**
- `pkg/agent/memory/unified_memory.go`
- `pkg/agent/memory/qdrant_store.go`

**Implementation Details:**

1. Add batch operations to `qdrant_store.go`:

```go
// BatchStoreVectors stores multiple vectors in a single operation
func (q *QDrantStore) BatchStoreVectors(ctx context.Context, vectors map[string][]float32, payloads map[string]map[string]interface{}) error {
    if len(vectors) == 0 {
        return nil
    }
    
    points := make([]*qdrant.PointStruct, 0, len(vectors))
    
    for id, vector := range vectors {
        payload, ok := payloads[id]
        if !ok {
            payload = make(map[string]interface{})
        }
        
        // Pre-process payload to handle time.Time values
        processedPayload := make(map[string]interface{})
        for k, v := range payload {
            switch val := v.(type) {
            case time.Time:
                processedPayload[k] = val.Format(time.RFC3339)
            case map[string]string:
                interfaceMap := make(map[string]interface{})
                for sk, sv := range val {
                    interfaceMap[sk] = sv
                }
                processedPayload[k] = interfaceMap
            default:
                processedPayload[k] = v
            }
        }
        
        points = append(points, &qdrant.PointStruct{
            Id:      qdrant.NewIDUUID(id),
            Vectors: qdrant.NewVectors(vector...),
            Payload: qdrant.NewValueMap(processedPayload),
        })
    }
    
    // Batch insert up to 100 points at a time
    for i := 0; i < len(points); i += 100 {
        end := i + 100
        if end > len(points) {
            end = len(points)
        }
        
        batchPoints := points[i:end]
        _, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{
            CollectionName: q.collection,
            Points:         batchPoints,
        })
        
        if err != nil {
            return fmt.Errorf("failed to batch store vectors in QDrant: %w", err)
        }
    }
    
    return nil
}

// BatchSearch performs multiple searches efficiently
func (q *QDrantStore) BatchSearch(ctx context.Context, queries [][]float32, limit int, filters map[string]interface{}) ([][]SearchResult, error) {
    if len(queries) == 0 {
        return nil, nil
    }
    
    results := make([][]SearchResult, len(queries))
    
    // For smaller batch sizes, use parallel individual queries
    if len(queries) < 10 {
        var wg sync.WaitGroup
        var mu sync.Mutex
        errChan := make(chan error, len(queries))
        
        for i, query := range queries {
            wg.Add(1)
            go func(idx int, qvec []float32) {
                defer wg.Done()
                
                searchResults, err := q.Search(ctx, qvec, limit, filters)
                if err != nil {
                    errChan <- err
                    return
                }
                
                mu.Lock()
                results[idx] = searchResults
                mu.Unlock()
            }(i, query)
        }
        
        wg.Wait()
        close(errChan)
        
        if err := <-errChan; err != nil {
            return nil, err
        }
        
        return results, nil
    }
    
    // For larger batches, we could use a bulk search API if available
    // Future implementation would depend on QDrant's batch search capability
    
    // Fallback to sequential search for now
    for i, query := range queries {
        searchResults, err := q.Search(ctx, query, limit, filters)
        if err != nil {
            return nil, err
        }
        results[i] = searchResults
    }
    
    return results, nil
}
```

2. Add batch memory operations to `unified_memory.go`:

```go
// BatchStoreMemories stores multiple memories efficiently in a single operation
func (um *UnifiedMemory) BatchStoreMemories(
    ctx context.Context,
    memories []struct {
        Content string
        AgentID string
        Type    MemoryType
        Source  string
        Metadata map[string]string
    },
) ([]string, error) {
    if len(memories) == 0 {
        return nil, nil
    }
    
    output.Verbose(fmt.Sprintf("Batch storing %d memories", len(memories)))
    
    // Get embeddings for all contents at once
    contents := make([]string, len(memories))
    for i, mem := range memories {
        contents[i] = mem.Content
    }
    
    var embeddings [][]float32
    if um.embeddingProvider != nil {
        var err error
        embeddings, err = um.BatchGetEmbeddings(ctx, contents)
        if err != nil {
            output.Warn(fmt.Sprintf("Failed to get batch embeddings: %v", err))
            // Continue with empty embeddings
        }
    }
    
    // Prepare memory entries and vector batch data
    memoryIDs := make([]string, len(memories))
    memoryEntries := make([]*EnhancedMemoryEntry, len(memories))
    vectorMap := make(map[string][]float32)
    payloadMap := make(map[string]map[string]interface{})
    
    for i, mem := range memories {
        // Generate a unique ID
        memoryID := uuid.New().String()
        memoryIDs[i] = memoryID
        
        // Prepare the memory entry
        entry := &EnhancedMemoryEntry{
            ID:          memoryID,
            AgentID:     mem.AgentID,
            Content:     mem.Content,
            Type:        mem.Type,
            Source:      mem.Source,
            CreatedAt:   time.Now(),
            AccessCount: 0,
            LastAccess:  time.Now(),
            Metadata:    mem.Metadata,
        }
        
        // Add embedding if available
        if i < len(embeddings) {
            entry.Embedding = embeddings[i]
            
            // Prepare vector data for batch storage
            vectorMap[memoryID] = embeddings[i]
            
            // Prepare payload
            payload := map[string]interface{}{
                "agent_id":   mem.AgentID,
                "content":    mem.Content,
                "type":       string(mem.Type),
                "source":     mem.Source,
                "created_at": entry.CreatedAt,
                "metadata":   mem.Metadata,
            }
            payloadMap[memoryID] = payload
        }
        
        memoryEntries[i] = entry
    }
    
    // Store all entries in memory
    um.mutex.Lock()
    for _, entry := range memoryEntries {
        um.memoryData[entry.ID] = entry
    }
    um.mutex.Unlock()
    
    // Store in vector store (batch operation)
    if um.options.EnableVectorStore && um.vectorStore != nil && len(vectorMap) > 0 {
        vectorStoreSpinner := output.StartSpinner("Batch storing in vector database")
        
        if store, ok := um.vectorStore.(*QDrantStore); ok {
            err := store.BatchStoreVectors(ctx, vectorMap, payloadMap)
            if err != nil {
                output.StopSpinner(vectorStoreSpinner, "")
                output.Error("Failed to batch store in vector store", err)
            } else {
                output.StopSpinner(vectorStoreSpinner, fmt.Sprintf("Stored %d memories in vector database", len(vectorMap)))
            }
        } else {
            // Fallback to individual storage
            output.StopSpinner(vectorStoreSpinner, "")
            output.Warn("Batch storage not supported by vector store implementation")
            
            for id, vector := range vectorMap {
                err := um.vectorStore.StoreVector(ctx, id, vector, payloadMap[id])
                if err != nil {
                    output.Error(fmt.Sprintf("Failed to store memory %s in vector store", id), err)
                }
            }
        }
    }
    
    // Store in graph store (future batch implementation)
    if um.options.EnableGraphStore && um.graphStore != nil {
        // For now, call individual operations
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
            
            err := um.graphStore.CreateNode(ctx, entry.ID, labels, properties)
            if err != nil {
                output.Error(fmt.Sprintf("Failed to store memory %s in graph store", entry.ID), err)
            }
        }
    }
    
    return memoryIDs, nil
}

// BatchGetEmbeddings efficiently gets embeddings for multiple texts
func (um *UnifiedMemory) BatchGetEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
    if um.embeddingProvider == nil {
        return nil, errors.New("no embedding provider available")
    }
    
    // If the provider supports batch operations, use it
    if batchProvider, ok := um.embeddingProvider.(BatchEmbeddingProvider); ok {
        return batchProvider.GetBatchEmbeddings(ctx, texts)
    }
    
    // Otherwise, fallback to sequential operation
    embeddings := make([][]float32, len(texts))
    var wg sync.WaitGroup
    var mu sync.Mutex
    errChan := make(chan error, len(texts))
    
    for i, text := range texts {
        if text == "" {
            continue
        }
        
        wg.Add(1)
        go func(idx int, content string) {
            defer wg.Done()
            
            embedding, err := um.embeddingProvider.GetEmbedding(ctx, content)
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
    
    if err := <-errChan; err != nil {
        return embeddings, err
    }
    
    return embeddings, nil
}

// BatchEmbeddingProvider defines interface for providers supporting batch operations
type BatchEmbeddingProvider interface {
    EmbeddingProvider
    GetBatchEmbeddings(ctx context.Context, texts []string) ([][]float32, error)
}
```
