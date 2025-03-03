package tools

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"google.golang.org/api/customsearch/v1"
	"google.golang.org/api/option"
)

// WebSearch is a tool for performing web searches
type WebSearch struct {
	APIKey     string
	SearchID   string
	BaseURL    string
	MaxResults int
}

// NewWebSearch creates a new WebSearch tool
func NewWebSearch(apiKey, searchID string) *WebSearch {
	return &WebSearch{
		APIKey:     apiKey,
		SearchID:   searchID,
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
func (w *WebSearch) Execute(ctx context.Context, args map[string]any) (any, error) {
	query, ok := args["query"].(string)
	if !ok {
		return nil, errors.New("query argument is required")
	}

	svc, err := customsearch.NewService(ctx, option.WithAPIKey(os.Getenv("GOOGLE_API_KEY")))
	if err != nil {
		return nil, err
	}

	resp, err := svc.Cse.List().Cx(w.SearchID).Q(query).Do()
	if err != nil {
		return nil, err
	}

	var results strings.Builder

	for i, result := range resp.Items {
		if i >= w.MaxResults {
			break
		}
		results.WriteString(fmt.Sprintf("#%d: %s\n", i+1, result.Title))
		results.WriteString(fmt.Sprintf("\t%s\n", result.Snippet))
		results.WriteString(fmt.Sprintf("\t%s\n", result.Link))
		results.WriteString("\n")
	}

	return results.String(), nil
}

// Schema returns the JSON schema for the tool's arguments
func (w *WebSearch) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "The search query",
			},
		},
		"required": []string{"query"},
	}
}
