package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/errnie"
)

// MemoryExtractionResult represents the JSON response from the LLM
type MemoryExtractionResult struct {
	Memories []struct {
		Content     string  `json:"content"`
		Importance  float64 `json:"importance"`
		Explanation string  `json:"explanation"`
	} `json:"memories"`
}

// LLMMemoryExtractor uses an LLM to extract memories from text
type LLMMemoryExtractor struct {
	BaseMemory
	llmProvider core.LLMProvider
	prompt      string
	apiKey      string // For OpenAI API direct calls
	model       string // For OpenAI API direct calls
}

// NewLLMMemoryExtractor creates a new LLM-based memory extractor with a core.LLMProvider
func NewLLMMemoryExtractor(llmProvider core.LLMProvider) *LLMMemoryExtractor {
	defaultPrompt := createDefaultExtractionPrompt()
	return &LLMMemoryExtractor{
		BaseMemory:  *NewBaseMemory(),
		llmProvider: llmProvider,
		prompt:      defaultPrompt,
	}
}

// NewOpenAIMemoryExtractor creates a new memory extractor using direct OpenAI API calls
func NewOpenAIMemoryExtractor(apiKey, model string) *LLMMemoryExtractor {
	if model == "" {
		model = "gpt-4o-mini" // Default model for extraction
	}

	defaultPrompt := createDefaultExtractionPrompt()
	return &LLMMemoryExtractor{
		BaseMemory: *NewBaseMemory(),
		apiKey:     apiKey,
		model:      model,
		prompt:     defaultPrompt,
	}
}

// createDefaultExtractionPrompt creates the default extraction prompt template
func createDefaultExtractionPrompt() string {
	return `You are a memory extraction system for an agent. Your job is to extract important information from the conversation that might be useful to remember for future interactions.

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
}

// SetPrompt allows customizing the extraction prompt
func (e *LLMMemoryExtractor) SetPrompt(prompt string) {
	e.prompt = prompt
}

// ExtractMemories processes text to extract important memories using an LLM
func (e *LLMMemoryExtractor) ExtractMemories(ctx context.Context, agentName, text, source string) ([]core.MemoryEntry, error) {
	// If we have an LLM provider, use it; otherwise use direct API calls
	if e.llmProvider != nil {
		return e.extractWithLLMProvider(ctx, agentName, text, source)
	} else if e.apiKey != "" {
		return e.extractWithOpenAIAPI(ctx, agentName, text, source)
	}

	// Fall back to base implementation if no LLM available
	return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
}

// extractWithLLMProvider extracts memories using the core.LLMProvider interface
func (e *LLMMemoryExtractor) extractWithLLMProvider(ctx context.Context, agentName, text, source string) ([]core.MemoryEntry, error) {
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

	return e.processExtractedMemories(ctx, agentName, source, result)
}

// extractWithOpenAIAPI extracts memories using direct OpenAI API calls
func (e *LLMMemoryExtractor) extractWithOpenAIAPI(ctx context.Context, agentName, text, source string) ([]core.MemoryEntry, error) {
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	type OpenAIRequest struct {
		Model     string    `json:"model"`
		Messages  []Message `json:"messages"`
		MaxTokens int       `json:"max_tokens"`
	}

	type OpenAIResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	// Prepare the system prompt for memory extraction
	systemPrompt := strings.Replace(e.prompt, "{{TEXT}}", "", 1)
	userPrompt := text

	// Build the request
	reqBody := OpenAIRequest{
		Model: e.model,
		Messages: []Message{
			{
				Role:    "system",
				Content: systemPrompt,
			},
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
		MaxTokens: 1000,
	}

	// Convert to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to marshal request: %v", err))
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to create HTTP request: %v", err))
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", e.apiKey))

	// Execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to execute HTTP request: %v", err))
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}
	defer resp.Body.Close()

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to read response body: %v", err))
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	if resp.StatusCode != http.StatusOK {
		errnie.Warn(fmt.Sprintf("API error: %s, status code: %d", string(body), resp.StatusCode))
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	// Parse the response
	var response OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		errnie.Warn(fmt.Sprintf("Failed to unmarshal response: %v", err))
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	if len(response.Choices) == 0 {
		errnie.Warn("No choices returned from API")
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	// Extract JSON content from the response
	jsonStr := extractJSON(response.Choices[0].Message.Content)
	if jsonStr == "" {
		errnie.Warn("Failed to find JSON in API response")
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	// Parse the JSON response
	var result MemoryExtractionResult
	err = json.Unmarshal([]byte(jsonStr), &result)
	if err != nil {
		errnie.Warn(fmt.Sprintf("Failed to parse memory extraction JSON: %v", err))
		return e.BaseMemory.ExtractMemories(ctx, agentName, text, source)
	}

	return e.processExtractedMemories(ctx, agentName, source, result)
}

// processExtractedMemories processes and stores the extracted memories
func (e *LLMMemoryExtractor) processExtractedMemories(ctx context.Context, agentName, source string, result MemoryExtractionResult) ([]core.MemoryEntry, error) {
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

// GenerateMemoryQueries generates query strings to search for relevant memories
func GenerateMemoryQueries(ctx context.Context, apiKey, model, query string) ([]string, error) {
	if model == "" {
		model = "gpt-4o-mini"
	}

	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	type OpenAIRequest struct {
		Model     string    `json:"model"`
		Messages  []Message `json:"messages"`
		MaxTokens int       `json:"max_tokens"`
	}

	type OpenAIResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	// Prepare the system prompt for query generation
	systemPrompt := `You are an AI memory query generator. Your task is to analyze the user's input and generate multiple search queries that would help retrieve relevant memories from a vector database.

Generate 3-5 diverse search queries that capture different aspects or interpretations of the user's input.
Format your response as a JSON array of strings, where each string is a distinct search query.
Make the queries specific enough to retrieve relevant information but general enough to capture related concepts.

Example response format:
["Query 1", "Query 2", "Query 3"]

Do not include any explanations, just the JSON array.`

	// Prepare the user prompt with the query to analyze
	userPrompt := fmt.Sprintf("Please generate memory search queries for the following input:\n\n%s", query)

	// Build the request
	reqBody := OpenAIRequest{
		Model: model,
		Messages: []Message{
			{
				Role:    "system",
				Content: systemPrompt,
			},
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
		MaxTokens: 300,
	}

	// Convert to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	// Execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: %s, status code: %d", string(body), resp.StatusCode)
	}

	// Parse the response
	var response OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from API")
	}

	// Extract the queries from the response
	content := response.Choices[0].Message.Content

	// Try to parse as JSON array first
	var queries []string
	err = json.Unmarshal([]byte(content), &queries)
	if err == nil && len(queries) > 0 {
		return queries, nil
	}

	// If not valid JSON, try to extract JSON array using regex
	arrayRegex := regexp.MustCompile(`\[.*\]`)
	match := arrayRegex.FindString(content)

	if match != "" {
		// Try to parse the JSON array
		if err := json.Unmarshal([]byte(match), &queries); err == nil {
			return queries, nil
		}
	}

	// If still no valid JSON, try to parse line by line
	queries = []string{}
	lines := strings.Split(content, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		// Skip empty lines and lines that look like headers or explanations
		if line == "" || !strings.HasPrefix(line, "\"") && !strings.HasPrefix(line, "- ") {
			continue
		}

		// Remove any bullet points or quotes
		line = strings.TrimPrefix(line, "- ")
		line = strings.Trim(line, "\"'")

		if line != "" {
			queries = append(queries, line)
		}
	}

	if len(queries) > 0 {
		return queries, nil
	}

	// If all else fails, just use the original query
	return []string{query}, nil
}
