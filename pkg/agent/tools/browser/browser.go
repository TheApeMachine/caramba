package browser

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/output"
)

// Tool represents the Browser Tool for web interactions
type Tool struct {
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
func (t *Tool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"navigate", "screenshot", "pdf", "extract", "execute", "search"},
				"description": "Action to perform with the browser tool",
			},
			"url": map[string]interface{}{
				"type":        "string",
				"description": "URL to navigate to (required for navigate, screenshot, pdf, extract, optional for execute)",
			},
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Search query (required for search action)",
			},
			"selector": map[string]interface{}{
				"type":        "string",
				"description": "CSS selector for content extraction (required for extract)",
			},
			"script": map[string]interface{}{
				"type":        "string",
				"description": "JavaScript code to execute (required for execute)",
			},
			"wait_for": map[string]interface{}{
				"type":        "string",
				"description": "CSS selector to wait for before performing the action",
			},
			"full_page": map[string]interface{}{
				"type":        "boolean",
				"description": "Capture full page for screenshot/PDF (default: false)",
			},
			"timeout": map[string]interface{}{
				"type":        "integer",
				"description": "Operation timeout in seconds (default: 30)",
			},
		},
		"required": []string{"action"},
	}
}

// Execute executes the browser tool with the given action and arguments
func (t *Tool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, fmt.Errorf("action parameter is required")
	}

	output.Verbose(fmt.Sprintf("Executing browser tool action: %s", action))

	// Create a spinner for long-running operations
	spinner := output.StartSpinner(fmt.Sprintf("Browser %s in progress...", action))
	defer output.StopSpinner(spinner, fmt.Sprintf("Browser %s completed", action))

	var result interface{}
	var err error

	switch action {
	case "navigate":
		result, err = t.navigate(ctx, args)
	case "screenshot":
		result, err = t.screenshot(ctx, args)
	case "extract":
		result, err = t.extractContent(ctx, args)
	case "search":
		result, err = t.search(ctx, args)
	case "pdf":
		result, err = t.generatePDF(ctx, args)
	case "execute":
		result, err = t.executeScript(ctx, args)
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}

	if err != nil {
		output.Error(fmt.Sprintf("Browser %s failed: %v", action, err), err)
		return nil, err
	}

	return result, nil
}

// createRequestWithAuth creates a new HTTP request with the API key
func (t *Tool) createRequestWithAuth(method, url string, body []byte) (*http.Request, error) {
	req, err := http.NewRequest(method, url, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
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
func (t *Tool) sendRequest(ctx context.Context, req *http.Request) (map[string]interface{}, error) {
	// Use the context from the request
	reqWithContext := req.WithContext(ctx)

	// Send the request
	resp, err := t.client.Do(reqWithContext)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check for error status codes
	if resp.StatusCode >= 400 {
		var errorBody map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&errorBody); err == nil {
			if message, ok := errorBody["message"].(string); ok {
				return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, message)
			}
		}
		return nil, fmt.Errorf("API error (status %d)", resp.StatusCode)
	}

	// Parse the response
	var result map[string]interface{}

	// For some endpoints (like PDF), the response might be binary data
	if strings.Contains(resp.Header.Get("Content-Type"), "application/pdf") {
		// For binary data, just set a placeholder in the result
		result = map[string]interface{}{
			"status": "success",
			"data":   resp.Body,
		}
	} else {
		// For JSON data, decode it
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return result, nil
}

// AvailableActions returns a list of available actions for the browser tool
func (t *Tool) AvailableActions() []map[string]string {
	return []map[string]string{
		{
			"name":        "navigate",
			"description": "Navigate to a URL and get the page HTML",
			"example":     `{"url": "https://example.com"}`,
		},
		{
			"name":        "screenshot",
			"description": "Take a screenshot of a webpage",
			"example":     `{"url": "https://example.com", "full_page": true}`,
		},
		{
			"name":        "extract",
			"description": "Extract content from a webpage using a CSS selector",
			"example":     `{"url": "https://example.com", "selector": ".main-content"}`,
		},
		{
			"name":        "search",
			"description": "Perform a web search with the given query",
			"example":     `{"query": "latest AI research papers"}`,
		},
		{
			"name":        "pdf",
			"description": "Generate a PDF of a webpage",
			"example":     `{"url": "https://example.com", "full_page": true}`,
		},
		{
			"name":        "execute",
			"description": "Execute JavaScript on a webpage",
			"example":     `{"url": "https://example.com", "script": "return document.title;"}`,
		},
	}
}
