package memory

import (
	"context"
	"fmt"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/errnie"
)

// QDrantMemory implements the Memory interface using QDrant vector database
type QDrantMemory struct {
	BaseMemory
	client         *qdrant.Client
	embeddings     EmbeddingsProvider
	collectionName string
	dimension      uint64 // Vector dimension
}

// QDrantConfig holds configuration for the QDrant client
type QDrantConfig struct {
	Host           string
	Port           int // Changed to int to match qdrant.Config
	APIKey         string
	UseTLS         bool
	CollectionName string
}

// NewQDrantMemory creates a new QDrantMemory instance
func NewQDrantMemory(embeddings EmbeddingsProvider, config QDrantConfig) (*QDrantMemory, error) {
	if config.CollectionName == "" {
		config.CollectionName = "long-term-memory"
	}

	// Create a new QDrant client
	clientConfig := &qdrant.Config{
		Host:   config.Host,
		Port:   config.Port,
		APIKey: config.APIKey,
		UseTLS: config.UseTLS,
	}

	client, err := qdrant.NewClient(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create QDrant client: %w", err)
	}

	// Get a sample vector to determine dimension
	// This is a bit of a hack but we need to know the dimension for collection creation
	sampleText := "Sample text to determine embedding dimension"
	sampleEmbedding, err := embeddings.GetEmbeddings(context.Background(), sampleText)
	if err != nil {
		return nil, fmt.Errorf("failed to generate sample embedding: %w", err)
	}

	dimension := uint64(len(sampleEmbedding))

	memory := &QDrantMemory{
		BaseMemory:     *NewBaseMemory(),
		client:         client,
		embeddings:     embeddings,
		collectionName: config.CollectionName,
		dimension:      dimension,
	}

	// Ensure collection exists
	err = memory.ensureCollection(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to ensure collection exists: %w", err)
	}

	return memory, nil
}

// ensureCollection creates the collection if it doesn't exist
func (m *QDrantMemory) ensureCollection(ctx context.Context) error {
	// Check if collection exists
	collections, err := m.client.ListCollections(ctx)
	if err != nil {
		return fmt.Errorf("failed to list collections: %w", err)
	}

	collectionExists := false
	for _, collectionName := range collections {
		if collectionName == m.collectionName {
			collectionExists = true
			break
		}
	}

	// Create collection if it doesn't exist
	if !collectionExists {
		err := m.client.CreateCollection(ctx, &qdrant.CreateCollection{
			CollectionName: m.collectionName,
			VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
				Size:     m.dimension,
				Distance: qdrant.Distance_Cosine,
			}),
		})
		if err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
		errnie.Info(fmt.Sprintf("Created QDrant collection: %s", m.collectionName))
	}

	return nil
}

// Store implements the Memory interface Store method with vector embeddings
func (m *QDrantMemory) Store(ctx context.Context, key string, value string) error {
	// First store in the base memory
	err := m.BaseMemory.Store(ctx, key, value)
	if err != nil {
		return err
	}

	// Generate embeddings for the value
	embeddings, err := m.embeddings.GetEmbeddings(ctx, value)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to generate embeddings for memory: %v", err))
		return nil // Still return success since the base memory storage worked
	}

	// Create point payload with the memory key
	payload := make(map[string]*qdrant.Value)
	payload["memory_key"] = &qdrant.Value{
		Kind: &qdrant.Value_StringValue{
			StringValue: key,
		},
	}
	payload["content"] = &qdrant.Value{
		Kind: &qdrant.Value_StringValue{
			StringValue: value,
		},
	}

	// Upsert the point into QDrant
	point := &qdrant.PointStruct{
		Id: &qdrant.PointId{
			PointIdOptions: &qdrant.PointId_Uuid{
				Uuid: key,
			},
		},
		Vectors: qdrant.NewVectors(embeddings...),
		Payload: payload,
	}

	_, err = m.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: m.collectionName,
		Points:         []*qdrant.PointStruct{point},
	})

	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to store vector in QDrant: %v", err))
		return nil // Still return success since the base memory storage worked
	}

	errnie.Info(fmt.Sprintf("Stored memory with QDrant embedding: %s", key))
	return nil
}

// Search implements semantic search using Qdrant vector search
func (m *QDrantMemory) Search(ctx context.Context, query string, limit int) ([]core.MemoryEntry, error) {
	// Generate embeddings for the query
	queryEmbeddings, err := m.embeddings.GetEmbeddings(ctx, query)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to generate embeddings for search query: %v", err))
		return m.BaseMemory.Search(ctx, query, limit) // Fall back to base implementation
	}

	// Create limit pointer
	limitUint64 := uint64(limit)

	// Query points using the Qdrant client
	searchRequest := &qdrant.QueryPoints{
		CollectionName: m.collectionName,
		Query:          qdrant.NewQuery(queryEmbeddings...),
		Limit:          &limitUint64,
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Enable{
				Enable: true,
			},
		},
	}

	searchResults, err := m.client.Query(ctx, searchRequest)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to search QDrant: %v", err))
		return m.BaseMemory.Search(ctx, query, limit) // Fall back to base implementation
	}

	// Convert search results to memory entries
	var entries []core.MemoryEntry
	for _, point := range searchResults {
		// Extract memory key from payload
		memoryKeyValue, ok := point.Payload["memory_key"]
		if !ok {
			continue
		}

		stringValue, ok := memoryKeyValue.Kind.(*qdrant.Value_StringValue)
		if !ok {
			continue
		}

		// Extract content from payload or use base memory retrieval
		var content interface{}
		contentValue, ok := point.Payload["content"]
		if ok {
			contentStr, ok := contentValue.Kind.(*qdrant.Value_StringValue)
			if ok {
				content = contentStr.StringValue
			}
		}

		// If content not in payload, get from base memory
		if content == nil {
			var retrieveErr error
			content, retrieveErr = m.BaseMemory.Retrieve(ctx, stringValue.StringValue)
			if retrieveErr != nil {
				continue
			}
		}

		entries = append(entries, core.MemoryEntry{
			Key:   stringValue.StringValue,
			Value: content,
			Score: float64(point.Score),
		})
	}

	return entries, nil
}

// ExtractMemories processes text to extract important memories and store with embeddings
func (m *QDrantMemory) ExtractMemories(ctx context.Context, agentName, text, source string) ([]core.MemoryEntry, error) {
	// For now, we'll just store the entire text as a single memory with embeddings
	timestamp := time.Now().UnixNano()
	key := fmt.Sprintf("%s_%s_%d", agentName, source, timestamp)

	err := m.Store(ctx, key, text)
	if err != nil {
		return nil, err
	}

	// Default score if we can't get embeddings
	score := 0.8

	entry := core.MemoryEntry{
		Key:   key,
		Value: text,
		Score: float64(score),
	}

	errnie.Info(fmt.Sprintf("Stored memory with QDrant embedding: %s", key))
	return []core.MemoryEntry{entry}, nil
}

// Clear implements the Memory interface
func (m *QDrantMemory) Clear(ctx context.Context) error {
	// Clear the base memory
	err := m.BaseMemory.Clear(ctx)
	if err != nil {
		return err
	}

	// Delete the collection
	err = m.client.DeleteCollection(ctx, m.collectionName)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to delete QDrant collection: %v", err))
	}

	// Recreate the collection
	err = m.ensureCollection(ctx)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to recreate QDrant collection: %v", err))
	}

	return nil
}
