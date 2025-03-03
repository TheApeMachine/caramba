package browser

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// Tool represents the Browser Tool for web interactions
type Tool struct {
	logger     *output.Logger
	hub        *hub.Queue
	apiKey     string
	apiBaseURL string
	client     *http.Client
	timeout    int // Default timeout in seconds
}

// New creates a new BrowserTool instance
func New(browserlessURL string, browserlessAPIKey string) *Tool {
	apiKey := browserlessAPIKey
	apiBaseURL := browserlessURL

	// Set a default if not provided
	if apiBaseURL == "" {
		apiBaseURL = "https://chrome.browserless.io"
	}

	// Create an HTTP client with the desired settings
	client := &http.Client{
		Timeout: 60 * time.Second, // Default timeout is 60 seconds for the client
	}

	return &Tool{
		logger:     output.NewLogger(),
		hub:        hub.NewQueue(),
		apiKey:     apiKey,
		apiBaseURL: apiBaseURL,
		client:     client,
		timeout:    30, // Default timeout in seconds for operations
	}
}

// Name returns the name of the tool
func (t *Tool) Name() string {
	return "browser"
}

// Description returns the description of the tool
func (t *Tool) Description() string {
	return "A tool for interacting with web pages, including navigation, taking screenshots, and extracting content."
}

// Schema returns the schema for the tool in JSON format
func (t *Tool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"action": map[string]any{
				"type":        "string",
				"enum":        []string{"navigate", "screenshot", "pdf", "extract", "execute", "search"},
				"description": "Action to perform with the browser tool",
			},
			"url": map[string]any{
				"type":        "string",
				"description": "URL to navigate to (required for navigate, screenshot, pdf, extract, optional for execute)",
			},
			"selector": map[string]any{
				"type":        "string",
				"description": "CSS selector for content extraction (required for extract)",
			},
			"script": map[string]any{
				"type":        "string",
				"description": "JavaScript code to execute (required for execute)",
			},
			"wait_for": map[string]any{
				"type":        "string",
				"description": "CSS selector to wait for before performing the action",
			},
			"full_page": map[string]any{
				"type":        "boolean",
				"description": "Capture full page for screenshot/PDF (default: false)",
			},
			"timeout": map[string]any{
				"type":        "integer",
				"description": "Operation timeout in seconds (default: 30)",
			},
		},
		"required": []string{"action"},
	}
}

// Execute executes the browser tool with the given action and arguments
func (t *Tool) Execute(ctx context.Context, args map[string]any) (any, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("action parameter is required"))
	}

	var result any
	var err error

	switch action {
	case "navigate":
		result, err = t.navigate(ctx, args)
	case "screenshot":
		result, err = t.screenshot(ctx, args)
	case "extract":
		result, err = t.extractContent(ctx, args)
	case "pdf":
		result, err = t.generatePDF(ctx, args)
	case "execute":
		result, err = t.executeScript(ctx, args)
	default:
		return nil, t.logger.Error(t.Name(), fmt.Errorf("unknown action: %s", action))
	}

	if err != nil {
		return nil, t.logger.Error(t.Name(), err)
	}

	return result, nil
}

// createRequestWithAuth creates a new HTTP request with the API key
func (t *Tool) createRequestWithAuth(method, url string, body []byte) (*http.Request, error) {
	req, err := http.NewRequest(method, url, strings.NewReader(string(body)))
	if err != nil {
		return nil, t.logger.Error(t.Name(), err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Cache-Control", "no-cache")

	// Add API key as query parameter
	q := req.URL.Query()
	q.Add("token", t.apiKey)
	req.URL.RawQuery = q.Encode()

	return req, nil
}

// sendRequest sends a request and processes the response
func (t *Tool) sendRequest(ctx context.Context, req *http.Request) (map[string]any, error) {
	// Use the context from the request
	reqWithContext := req.WithContext(ctx)

	// Send the request
	resp, err := t.client.Do(reqWithContext)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to send request: %w", err))
	}
	defer resp.Body.Close()

	// Check for error status codes
	if resp.StatusCode >= 400 {
		var errorBody map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&errorBody); err == nil {
			if message, ok := errorBody["message"].(string); ok {
				return nil, t.logger.Error(t.Name(), fmt.Errorf("API error (status %d): %s", resp.StatusCode, message))
			}
		}
		return nil, t.logger.Error(t.Name(), fmt.Errorf("API error (status %d)", resp.StatusCode))
	}

	// Parse the response
	var result map[string]any

	// For some endpoints (like PDF), the response might be binary data
	if strings.Contains(resp.Header.Get("Content-Type"), "application/pdf") {
		// For binary data, just set a placeholder in the result
		result = map[string]any{
			"status": "success",
			"data":   resp.Body,
		}
	} else {
		// For JSON data, decode it
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to decode response: %w", err))
		}
	}

	return result, nil
}
