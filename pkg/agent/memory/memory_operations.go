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
)

// extractMemoriesFromText extracts memories from text using OpenAI.
// This function uses the OpenAI API to identify interesting memories from text.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - apiKey: The OpenAI API key
//   - agentID: The agent ID to associate with the memories
//   - text: The text to extract memories from
//   - source: The source of the memories (e.g., conversation)
//
// Returns:
//   - A slice of extracted memory strings
//   - An error if the operation fails, or nil on success
func extractMemoriesFromText(ctx context.Context, apiKey, agentID, text, source string) ([]string, error) {
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
	systemPrompt := `You are an AI memory extraction system. Your task is to identify important information from the text that might be useful to remember for future context. 
	
Extract only the most salient facts, insights, or details that should be stored as long-term memories.
Format your response as a JSON array of strings, where each string is a distinct memory.
Only include memories that are factual and meaningful - not opinions or subjective assessments unless they represent the speaker's viewpoint.

Example response format:
["Memory 1", "Memory 2", "Memory 3"]

Do not include any explanations, just the JSON array. Keep each memory concise and focused on a single piece of information.`

	// Prepare the user prompt with the text to analyze
	userPrompt := fmt.Sprintf("Please extract important memories from the following text:\n\n%s", text)

	// Build the request
	reqBody := OpenAIRequest{
		Model: "gpt-3.5-turbo",
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
		MaxTokens: 500,
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
		return nil, fmt.Errorf("OpenAI API error: %s, status code: %d", string(body), resp.StatusCode)
	}

	// Parse the response
	var response OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from OpenAI")
	}

	// Extract the memories from the response
	content := response.Choices[0].Message.Content
	// The content should be a JSON array of strings, but sometimes it might include extra text
	// Use a regex to extract just the JSON array part
	arrayRegex := regexp.MustCompile(`\[.*\]`)
	match := arrayRegex.FindString(content)

	if match == "" {
		// If no JSON array is found, try to parse line by line
		var memories []string
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
				memories = append(memories, line)
			}
		}

		return memories, nil
	}

	// Try to parse the JSON array
	var memories []string
	if err := json.Unmarshal([]byte(match), &memories); err != nil {
		return nil, fmt.Errorf("failed to parse memories from response: %w", err)
	}

	return memories, nil
}

// generateMemoryQueries generates query strings to search for relevant memories.
// This function uses the OpenAI API to generate search queries based on user input.
//
// Parameters:
//   - ctx: The context for the operation, which can be used for cancellation
//   - apiKey: The OpenAI API key
//   - query: The user query to generate memory search queries for
//
// Returns:
//   - A slice of generated query strings
//   - An error if the operation fails, or nil on success
func generateMemoryQueries(ctx context.Context, apiKey, query string) ([]string, error) {
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
		Model: "gpt-3.5-turbo",
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
		return nil, fmt.Errorf("OpenAI API error: %s, status code: %d", string(body), resp.StatusCode)
	}

	// Parse the response
	var response OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from OpenAI")
	}

	// Extract the queries from the response
	content := response.Choices[0].Message.Content
	// The content should be a JSON array of strings, but sometimes it might include extra text
	// Use a regex to extract just the JSON array part
	arrayRegex := regexp.MustCompile(`\[.*\]`)
	match := arrayRegex.FindString(content)

	if match == "" {
		// If no JSON array is found, try to parse line by line
		var queries []string
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

		return queries, nil
	}

	// Try to parse the JSON array
	var queries []string
	if err := json.Unmarshal([]byte(match), &queries); err != nil {
		return nil, fmt.Errorf("failed to parse queries from response: %w", err)
	}

	return queries, nil
}
