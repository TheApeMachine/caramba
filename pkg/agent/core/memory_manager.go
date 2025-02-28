package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// MemoryManager handles memory operations for agents.
type MemoryManager struct {
	hub     *hub.Queue
	logger  *output.Logger
	memory  Memory
	agentID string
}

// NewMemoryManager creates a new MemoryManager.
func NewMemoryManager(agentID string, memory Memory) *MemoryManager {
	return &MemoryManager{
		hub:     hub.NewQueue(),
		logger:  output.NewLogger(),
		memory:  memory,
		agentID: agentID,
	}
}

// SetMemory sets the memory system for the manager.
func (mm *MemoryManager) SetMemory(memory Memory) {
	mm.memory = memory
	mm.logger.Log(fmt.Sprintf("Set memory system for agent %s", mm.agentID))
}

// InjectMemories enhances a message with relevant memories.
func (mm *MemoryManager) InjectMemories(ctx context.Context, message LLMMessage) LLMMessage {
	enhancedMessage := message
	if mm.memory != nil {
		if memoryEnhancer, ok := mm.memory.(MemoryEnhancer); ok {
			enhancedContext, err := memoryEnhancer.PrepareContext(ctx, mm.agentID, message.Content)

			if err != nil {
				mm.hub.Add(hub.NewEvent(
					mm.agentID,
					"ui",
					"memory",
					hub.EventTypeError,
					err.Error(),
					map[string]string{},
				))
			}
			if enhancedContext != "" {
				mm.hub.Add(hub.NewEvent(
					mm.agentID,
					"ui",
					"memory",
					hub.EventTypeMetric,
					fmt.Sprintf("%d", len(enhancedContext)-len(message.Content)),
					map[string]string{},
				))
				enhancedMessage.Content = enhancedContext
			}
		} else {
			mm.logger.Log("Memory system does not support context enhancement")
		}
	} else {
		mm.logger.Log("No memory system available")
	}
	return enhancedMessage
}

// ExtractMemories extracts and stores memories from conversation text.
func (mm *MemoryManager) ExtractMemories(ctx context.Context, contextWindow string) {
	if mm.memory != nil {
		if memoryExtractor, ok := mm.memory.(MemoryExtractor); ok {
			mm.logger.Log("Extracting memories from conversation")

			memories, err := memoryExtractor.ExtractMemories(ctx, mm.agentID, contextWindow, "conversation")
			if err != nil {
				mm.logger.Log(fmt.Sprintf("Memory extraction failed: %v", err))
			} else if memories != nil {
				mm.logger.Log(fmt.Sprintf("Extracted %d memories", len(memories)))
			}
		} else {
			mm.logger.Log("Memory system does not support memory extraction")
		}
	}
}

// CollectConversation builds a string representation of a conversation from messages.
func (mm *MemoryManager) CollectConversation(messages []LLMMessage) string {
	var contextWindow strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			contextWindow.WriteString(fmt.Sprintf("User: %s\n", msg.Content))
		case "assistant":
			contextWindow.WriteString(fmt.Sprintf("Assistant: %s\n", msg.Content))
		}
	}
	return contextWindow.String()
}
