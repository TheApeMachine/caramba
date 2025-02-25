package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/errnie"
)

// LLMMemoryExtractor uses an LLM to extract memories from text
type LLMMemoryExtractor struct {
	BaseMemory
	llmProvider core.LLMProvider
	prompt      string
}

// MemoryExtractionResult represents the JSON response from the LLM
type MemoryExtractionResult struct {
	Memories []struct {
		Content     string  `json:"content"`
		Importance  float64 `json:"importance"`
		Explanation string  `json:"explanation"`
	} `json:"memories"`
}

// NewLLMMemoryExtractor creates a new LLM-based memory extractor
func NewLLMMemoryExtractor(llmProvider core.LLMProvider) *LLMMemoryExtractor {
	defaultPrompt := `You are a memory extraction system for an agent. Your job is to extract important information from the conversation that might be useful to remember for future interactions.

Extract 1-5 key pieces of information as memories, focusing on:
1. Facts about the user
2. User preferences
3. Important topics or questions
4. Action items or commitments
5. Context that would be valuable in future conversations

For each memory, provide:
1. The actual memory content (keep it concise)
2. An importance score (0.0-1.0)
3. A brief explanation of why this memory is valuable

The conversation to analyze is: {{TEXT}}

Format your response as JSON:
{
  "memories": [
    {
      "content": "User's name is John Smith",
      "importance": 0.9,
      "explanation": "The user's name is critical information for personalization"
    },
    {
      "content": "User prefers documentation in Markdown format",
      "importance": 0.7,
      "explanation": "This preference will help tailor future responses"
    }
  ]
}
`

	return &LLMMemoryExtractor{
		BaseMemory:  *NewBaseMemory(),
		llmProvider: llmProvider,
		prompt:      defaultPrompt,
	}
}

// SetPrompt allows customizing the extraction prompt
func (e *LLMMemoryExtractor) SetPrompt(prompt string) {
	e.prompt = prompt
}

// ExtractMemories processes text to extract important memories using an LLM
func (e *LLMMemoryExtractor) ExtractMemories(ctx context.Context, agentName, text, source string) ([]core.MemoryEntry, error) {
	// Replace the placeholder with the actual text
	prompt := strings.Replace(e.prompt, "{{TEXT}}", text, 1)

	// Setup options for LLM
	options := core.LLMOptions{
		Temperature: 0.7,  // Balance creativity and consistency
		MaxTokens:   1000, // Ensure we have enough tokens for the response
	}

	// Call the LLM to extract memories
	response, err := e.llmProvider.GenerateResponse(ctx, prompt, options)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to extract memories: %v", err))
		// Fall back to base implementation if LLM fails
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	// Parse the JSON response
	var result MemoryExtractionResult

	// Find the JSON part in the response (in case the LLM added any additional text)
	jsonStr := extractJSON(response)
	if jsonStr == "" {
		errnie.Warn("Failed to find JSON in LLM response for memory extraction")
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	err = json.Unmarshal([]byte(jsonStr), &result)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to parse memory extraction JSON: %v", err))
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	// Convert extracted memories to MemoryEntry objects and store them
	var entries []core.MemoryEntry
	timestamp := time.Now().UnixNano()

	for i, mem := range result.Memories {
		// Create a key with timestamp to ensure uniqueness
		key := fmt.Sprintf("%s_%s_%d_%d", agentName, source, timestamp, i)

		// Store the memory
		err := e.Store(ctx, key, mem.Content)
		if err != nil {
			errnie.Warn(fmt.Sprintf("Failed to store extracted memory: %v", err))
			continue
		}

		// Add to the list of entries
		entries = append(entries, core.MemoryEntry{
			Key:   key,
			Value: mem.Content,
			Score: mem.Importance,
		})

		errnie.Info(fmt.Sprintf("Extracted memory: %s (importance: %.2f)", mem.Content, mem.Importance))
	}

	return entries, nil
}

// extractJSON attempts to find and extract JSON from a string
func extractJSON(text string) string {
	// Look for JSON opening and closing braces
	start := strings.Index(text, "{")
	if start == -1 {
		return ""
	}

	// Find the matching closing brace
	depth := 0
	for i := start; i < len(text); i++ {
		if text[i] == '{' {
			depth++
		} else if text[i] == '}' {
			depth--
			if depth == 0 {
				return text[start : i+1]
			}
		}
	}

	return ""
}
