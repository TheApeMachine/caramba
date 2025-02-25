package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// WebSearch is a tool for performing web searches
type WebSearch struct {
	APIKey     string
	SearchID   string
	BaseURL    string
	MaxResults int
}

// SearchResult represents a search result
type SearchResult struct {
	Title       string `json:"title"`
	Link        string `json:"link"`
	Description string `json:"snippet"`
	Source      string `json:"source"`
}

// NewWebSearch creates a new WebSearch tool
func NewWebSearch(apiKey, searchID string) *WebSearch {
	return &WebSearch{
		APIKey:     apiKey,
		SearchID:   searchID,
		BaseURL:    "https://www.googleapis.com/customsearch/v1",
		MaxResults: 5,
	}
}

// Name returns the name of the tool
func (w *WebSearch) Name() string {
	return "web_search"
}

// Description returns the description of the tool
func (w *WebSearch) Description() string {
	return "Searches the web for information"
}

// Execute executes the tool with the given arguments
func (w *WebSearch) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query, ok := args["query"].(string)
	if !ok {
		return nil, errors.New("query must be a string")
	}
	
	if w.APIKey == "" || w.SearchID == "" {
		// Return mock results for development
		return w.getMockResults(query), nil
	}
	
	maxResults := w.MaxResults
	if maxResultsArg, ok := args["max_results"].(float64); ok {
		maxResults = int(maxResultsArg)
	}
	
	// Build the search URL
	params := url.Values{}
	params.Add("key", w.APIKey)
	params.Add("cx", w.SearchID)
	params.Add("q", query)
	params.Add("num", fmt.Sprintf("%d", maxResults))
	
	// Send the request
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, w.BaseURL+"?"+params.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search API error: %s, status: %d", string(body), resp.StatusCode)
	}
	
	var searchResponse struct {
		Items []struct {
			Title       string `json:"title"`
			Link        string `json:"link"`
			Snippet     string `json:"snippet"`
			DisplayLink string `json:"displayLink"`
		} `json:"items"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&searchResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	
	// Convert to SearchResult format
	results := make([]SearchResult, len(searchResponse.Items))
	for i, item := range searchResponse.Items {
		results[i] = SearchResult{
			Title:       item.Title,
			Link:        item.Link,
			Description: item.Snippet,
			Source:      item.DisplayLink,
		}
	}
	
	return results, nil
}

// Schema returns the JSON schema for the tool's arguments
func (w *WebSearch) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "The search query",
			},
			"max_results": map[string]interface{}{
				"type":        "number",
				"description": "Maximum number of results to return",
			},
		},
		"required": []string{"query"},
	}
}

// getMockResults returns mock search results for development
func (w *WebSearch) getMockResults(query string) []SearchResult {
	// This is a placeholder that should be replaced with actual search logic
	query = strings.ToLower(query)
	
	results := []SearchResult{
		{
			Title:       fmt.Sprintf("Information about %s - Wikipedia", query),
			Link:        fmt.Sprintf("https://en.wikipedia.org/wiki/%s", strings.ReplaceAll(query, " ", "_")),
			Description: fmt.Sprintf("This article is about %s. %s is a topic of interest to many people around the world.", query, strings.ToUpper(query[:1])+query[1:]),
			Source:      "en.wikipedia.org",
		},
		{
			Title:       fmt.Sprintf("Latest News on %s", query),
			Link:        fmt.Sprintf("https://news.example.com/%s", strings.ReplaceAll(query, " ", "-")),
			Description: fmt.Sprintf("Get the latest news about %s. Updates, developments, and expert analysis.", query),
			Source:      "news.example.com",
		},
		{
			Title:       fmt.Sprintf("%s - Official Website", query),
			Link:        fmt.Sprintf("https://%s.org", strings.ReplaceAll(query, " ", "")),
			Description: fmt.Sprintf("The official website for %s. Find resources, contact information, and more.", query),
			Source:      fmt.Sprintf("%s.org", strings.ReplaceAll(query, " ", "")),
		},
	}
	
	return results
}
