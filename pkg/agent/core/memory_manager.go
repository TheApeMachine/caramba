package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/errnie"
)

// MemoryManager handles memory operations for agents.
type MemoryManager struct {
	memory  Memory
	agentID string
}

// NewMemoryManager creates a new MemoryManager.
func NewMemoryManager(agentID string, memory Memory) *MemoryManager {
	return &MemoryManager{
		memory:  memory,
		agentID: agentID,
	}
}

// SetMemory sets the memory system for the manager.
func (mm *MemoryManager) SetMemory(memory Memory) {
	mm.memory = memory
	output.Verbose(fmt.Sprintf("Set memory system for agent %s", mm.agentID))
}

// InjectMemories enhances a message with relevant memories.
func (mm *MemoryManager) InjectMemories(ctx context.Context, message LLMMessage) LLMMessage {
	enhancedMessage := message
	if mm.memory != nil {
		if memoryEnhancer, ok := mm.memory.(MemoryEnhancer); ok {
			enhancedContext, err := memoryEnhancer.PrepareContext(ctx, mm.agentID, message.Content)
			if err == nil && enhancedContext != "" {
				output.Verbose(fmt.Sprintf("Enhanced input with memories (%d → %d chars)",
					len(message.Content), len(enhancedContext)))
				enhancedMessage.Content = enhancedContext
				errnie.Info(fmt.Sprintf("Enhanced input with %d characters of memories",
					len(enhancedContext)-len(message.Content)))
			} else if err != nil {
				output.Debug(fmt.Sprintf("Memory enhancement failed: %v", err))
			} else {
				output.Debug("No relevant memories found")
			}
		} else {
			output.Debug("Memory system does not support context enhancement")
		}
	} else {
		output.Debug("No memory system available")
	}
	return enhancedMessage
}

// ExtractMemories extracts and stores memories from conversation text.
func (mm *MemoryManager) ExtractMemories(ctx context.Context, contextWindow string) {
	if mm.memory != nil {
		if memoryExtractor, ok := mm.memory.(MemoryExtractor); ok {
			output.Verbose("Extracting memories from conversation")

			memories, err := memoryExtractor.ExtractMemories(ctx, mm.agentID, contextWindow, "conversation")
			if err != nil {
				output.Error("Memory extraction failed", err)
				errnie.Error(err)
			} else if memories != nil {
				output.Result(fmt.Sprintf("Extracted %d memories", len(memories)))
			}
		} else {
			output.Debug("Memory system does not support memory extraction")
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
