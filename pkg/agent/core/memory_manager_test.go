package core

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// MockMemoryEnhancer implements both Memory and MemoryEnhancer for testing
type MockMemoryEnhancer struct {
	MockMemory
	enhancedContext string
}

func (m *MockMemoryEnhancer) PrepareContext(ctx context.Context, agentID string, query string) (string, error) {
	return m.enhancedContext, nil
}

// MockMemoryExtractor implements both Memory and MemoryExtractor for testing
type MockMemoryExtractor struct {
	MockMemory
	extractedMemories []string
}

func (m *MockMemoryExtractor) ExtractMemories(ctx context.Context, agentID string, text string, source string) ([]string, error) {
	return m.extractedMemories, nil
}

func TestNewMemoryManager(t *testing.T) {
	Convey("Given a need for a memory manager", t, func() {
		Convey("When creating a new memory manager", func() {
			memory := &MockMemory{}
			manager := NewMemoryManager("test-agent", memory)

			Convey("Then it should not be nil", func() {
				So(manager, ShouldNotBeNil)
			})

			Convey("Then it should have the correct agent ID", func() {
				So(manager.agentID, ShouldEqual, "test-agent")
			})

			Convey("Then it should have the memory set", func() {
				So(manager.memory, ShouldEqual, memory)
			})
		})
	})
}

func TestSetMemory(t *testing.T) {
	Convey("Given a memory manager", t, func() {
		initialMemory := &MockMemory{}
		manager := NewMemoryManager("test-agent", initialMemory)

		Convey("When setting a new memory system", func() {
			newMemory := &MockMemory{}
			manager.SetMemory(newMemory)

			Convey("Then the memory should be updated", func() {
				So(manager.memory, ShouldEqual, newMemory)
			})
		})
	})
}

func TestInjectMemories(t *testing.T) {
	Convey("Given a memory manager with a memory enhancer", t, func() {
		enhancer := &MockMemoryEnhancer{enhancedContext: "Enhanced context with memories"}
		manager := NewMemoryManager("test-agent", enhancer)

		Convey("When injecting memories into a message", func() {
			message := LLMMessage{
				Role:    "user",
				Content: "Original query",
			}

			enhancedMessage := manager.InjectMemories(context.Background(), message)

			Convey("Then the message content should be enhanced", func() {
				So(enhancedMessage.Content, ShouldEqual, "Enhanced context with memories")
			})
		})

		Convey("When the memory system is not a memory enhancer", func() {
			manager.SetMemory(&MockMemory{})

			message := LLMMessage{
				Role:    "user",
				Content: "Original query",
			}

			enhancedMessage := manager.InjectMemories(context.Background(), message)

			Convey("Then the message content should remain unchanged", func() {
				So(enhancedMessage.Content, ShouldEqual, "Original query")
			})
		})

		Convey("When the memory system is nil", func() {
			manager.SetMemory(nil)

			message := LLMMessage{
				Role:    "user",
				Content: "Original query",
			}

			enhancedMessage := manager.InjectMemories(context.Background(), message)

			Convey("Then the message content should remain unchanged", func() {
				So(enhancedMessage.Content, ShouldEqual, "Original query")
			})
		})
	})
}

func TestExtractMemories(t *testing.T) {
	Convey("Given a memory manager with a memory extractor", t, func() {
		extractor := &MockMemoryExtractor{extractedMemories: []string{"Memory 1", "Memory 2"}}
		manager := NewMemoryManager("test-agent", extractor)

		Convey("When extracting memories from conversation text", func() {
			contextWindow := "User: What is the capital of France?\nAssistant: The capital of France is Paris."

			// This doesn't return anything, so we just verify it doesn't panic
			manager.ExtractMemories(context.Background(), contextWindow)

			Convey("Then it should complete without error", func() {
				// No assertion needed, just checking it doesn't panic
				So(true, ShouldBeTrue)
			})
		})

		Convey("When the memory system is not a memory extractor", func() {
			manager.SetMemory(&MockMemory{})

			contextWindow := "User: What is the capital of France?\nAssistant: The capital of France is Paris."

			// This should log a message but not panic
			manager.ExtractMemories(context.Background(), contextWindow)

			Convey("Then it should complete without error", func() {
				// No assertion needed, just checking it doesn't panic
				So(true, ShouldBeTrue)
			})
		})

		Convey("When the memory system is nil", func() {
			manager.SetMemory(nil)

			contextWindow := "User: What is the capital of France?\nAssistant: The capital of France is Paris."

			// This should log a message but not panic
			manager.ExtractMemories(context.Background(), contextWindow)

			Convey("Then it should complete without error", func() {
				// No assertion needed, just checking it doesn't panic
				So(true, ShouldBeTrue)
			})
		})
	})
}

func TestCollectConversation(t *testing.T) {
	Convey("Given a memory manager", t, func() {
		manager := NewMemoryManager("test-agent", &MockMemory{})

		Convey("When collecting conversation from messages", func() {
			messages := []LLMMessage{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there"},
				{Role: "user", Content: "How are you?"},
				{Role: "assistant", Content: "I'm doing well, thanks!"},
			}

			conversation := manager.CollectConversation(messages)

			Convey("Then it should format the conversation correctly", func() {
				expected := "User: Hello\nAssistant: Hi there\nUser: How are you?\nAssistant: I'm doing well, thanks!\n"
				So(conversation, ShouldEqual, expected)
			})
		})

		Convey("When collecting conversation from empty messages", func() {
			messages := []LLMMessage{}

			conversation := manager.CollectConversation(messages)

			Convey("Then it should return an empty string", func() {
				So(conversation, ShouldEqual, "")
			})
		})
	})
}
