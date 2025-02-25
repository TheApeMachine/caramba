package examples

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/memory"
	"github.com/theapemachine/errnie"
)

// MemoryExample demonstrates unified memory capabilities with vector and graph integration
func MemoryExample(apiKey string) error {
	fmt.Println("Running memory example...")

	// Create an LLM provider for the agent
	provider := llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")

	// Create an embedding provider for demonstration
	embeddingProvider := NewMockEmbeddingProvider()

	// Create base memory store
	baseMemory := memory.NewInMemoryStore()

	// Create memory options
	memoryOpts := memory.DefaultUnifiedMemoryOptions()
	memoryOpts.EnableVectorStore = true
	memoryOpts.EnableGraphStore = false // Using mock, so no need for real graph store

	// Create unified memory system
	unifiedMemory, err := memory.NewUnifiedMemory(baseMemory, embeddingProvider, memoryOpts)
	if err != nil {
		errnie.Error(err)
		return err
	}

	// Store some memories
	fmt.Println("\n=== Storing Memories ===")

	memories := []struct {
		content string
		source  string
		agentID string
		memType memory.MemoryType
	}{
		{"The capital of France is Paris.", "geography", "agent1", memory.MemoryTypePersonal},
		{"Python is a high-level programming language.", "programming", "agent1", memory.MemoryTypePersonal},
		{"Water boils at 100 degrees Celsius at sea level.", "science", "agent1", memory.MemoryTypePersonal},
		{"The Eiffel Tower is in Paris.", "geography", "agent1", memory.MemoryTypePersonal},
		{"JavaScript is used for web development.", "programming", "agent1", memory.MemoryTypePersonal},
	}

	for _, m := range memories {
		// Store each memory using regular key-value store
		memoryID := fmt.Sprintf("mem-%d", time.Now().UnixNano())
		err := unifiedMemory.Store(context.Background(), memoryID, m.content)

		// Also store using StoreMemory for vector storage
		_, err = unifiedMemory.StoreMemory(context.Background(), m.agentID, m.content, m.memType, m.source,
			map[string]interface{}{"category": m.source})

		if err != nil {
			errnie.Info(fmt.Sprintf("Failed to store memory: %v", err))
		}
		fmt.Printf("Stored: %s\n", m.content)
	}

	// Create relationships between memories
	fmt.Println("\n=== Creating Relationships ===")

	// Wait a moment for background processing
	time.Sleep(100 * time.Millisecond)

	// Search for some memories to get their IDs
	parisMemories, err := unifiedMemory.RetrieveMemoriesByVector(context.Background(), "Paris", "agent1", 2, 0.5)
	if err != nil {
		errnie.Info(fmt.Sprintf("Failed to search memories: %v", err))
	}

	programmingMemories, err := unifiedMemory.RetrieveMemoriesByVector(context.Background(), "programming", "agent1", 2, 0.5)
	if err != nil {
		errnie.Info(fmt.Sprintf("Failed to search memories: %v", err))
	}

	if len(parisMemories) > 0 && len(programmingMemories) > 0 {
		relation := memory.Relationship{
			FromID:   parisMemories[0].ID,
			ToID:     programmingMemories[0].ID,
			Type:     "related_to",
			Metadata: map[string]interface{}{"strength": 0.5},
		}

		err := unifiedMemory.CreateRelationship(context.Background(), relation)
		if err != nil {
			fmt.Printf("Note: Relationship creation failed: %v (expected if graph store is disabled)\n", err)
		} else {
			fmt.Printf("Created relationship between '%s' and '%s'\n",
				parisMemories[0].Content, programmingMemories[0].Content)
		}
	}

	// Demonstrate memory search
	fmt.Println("\n=== Memory Search ===")
	query := "Tell me about Paris"
	fmt.Printf("User query: %s\n", query)

	// Prepare context by extracting relevant memories
	enhancedContext, err := unifiedMemory.PrepareContext(context.Background(), "agent1", query)
	if err != nil {
		errnie.Error(err)
		return err
	}

	fmt.Println("Enhanced context for the query:")
	fmt.Println(enhancedContext)

	// Create an agent with the unified memory
	agent := core.NewAgentBuilder("MemoryAgent").
		WithLLM(provider).
		WithMemory(unifiedMemory).
		Build()

	// Execute the agent with the query
	fmt.Println("\n=== Agent Response ===")
	systemPrompt := "You are a helpful assistant. Use the provided memories to answer the user's question."
	fullPrompt := fmt.Sprintf("%s\n\n%s", systemPrompt, enhancedContext)

	resp, err := agent.Execute(context.Background(), fullPrompt)
	if err != nil {
		errnie.Error(err)
		return err
	}

	fmt.Printf("Agent response:\n%s\n", resp)

	return nil
}

// MockEmbeddingProvider is a simple mock implementation of the EmbeddingProvider interface
type MockEmbeddingProvider struct{}

// NewMockEmbeddingProvider creates a new mock embedding provider
func NewMockEmbeddingProvider() *MockEmbeddingProvider {
	return &MockEmbeddingProvider{}
}

// GetEmbedding returns a mock embedding
func (m *MockEmbeddingProvider) GetEmbedding(_ context.Context, text string) ([]float32, error) {
	// Create a simple deterministic embedding based on the text
	embedding := make([]float32, 5) // Small size for demonstration

	// Simple hash-like function to convert text to embedding
	var sum int
	for _, char := range text {
		sum += int(char)
	}

	// Fill the embedding with values derived from the text
	for i := range embedding {
		embedding[i] = float32(sum%(i+10)) / 10.0
	}

	return embedding, nil
}
