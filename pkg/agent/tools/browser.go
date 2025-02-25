package tools

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// BrowserTool provides a headless browser capability using Browserless
type BrowserTool struct {
	// browserlessURL is the URL of the Browserless instance
	browserlessURL string
	// token is the authentication token for Browserless
	token string
	// timeout for browser operations in seconds (applied client-side)
	timeout int
	// defaultUserAgent is the default User-Agent to use
	defaultUserAgent string
	// httpClient for making requests to Browserless
	httpClient *http.Client
}

// NewBrowserTool creates a new BrowserTool
func NewBrowserTool(browserlessURL, token string) *BrowserTool {
	if browserlessURL == "" {
		browserlessURL = "http://browserless:3000" // Default if using docker-compose
	}

	return &BrowserTool{
		browserlessURL:   browserlessURL,
		token:            token,
		timeout:          30,
		defaultUserAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
		httpClient: &http.Client{
			Timeout: time.Second * 60, // Longer timeout for the HTTP client
		},
	}
}

// Name returns the name of the tool
func (b *BrowserTool) Name() string {
	return "browser"
}

// Description returns the description of the tool
func (b *BrowserTool) Description() string {
	return "Controls a headless browser to visit websites, take screenshots, and extract content"
}

// Execute executes the tool with the given arguments
func (b *BrowserTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, errors.New("action must be a string")
	}

	switch action {
	case "navigate":
		return b.navigate(ctx, args)
	case "screenshot":
		return b.screenshot(ctx, args)
	case "pdf":
		return b.generatePDF(ctx, args)
	case "extract":
		return b.extractContent(ctx, args)
	case "execute":
		return b.executeScript(ctx, args)
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

// Schema returns the JSON schema for the tool's arguments
func (b *BrowserTool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"navigate", "screenshot", "pdf", "extract", "execute"},
				"description": "Action to perform (navigate, screenshot, pdf, extract, execute)",
			},
			"url": map[string]interface{}{
				"type":        "string",
				"description": "URL to navigate to (required for navigate, screenshot, pdf, extract)",
			},
			"selector": map[string]interface{}{
				"type":        "string",
				"description": "CSS selector to extract content from (required for extract)",
			},
			"script": map[string]interface{}{
				"type":        "string",
				"description": "JavaScript to execute (required for execute)",
			},
			"wait_for": map[string]interface{}{
				"type":        "string",
				"description": "CSS selector to wait for before taking action",
			},
			"fullPage": map[string]interface{}{
				"type":        "boolean",
				"description": "Whether to capture the full page (for screenshot/pdf)",
			},
		},
		"required": []string{"action"},
	}
}

// navigate navigates to a URL and returns the page HTML
func (b *BrowserTool) navigate(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, errors.New("url must be a string")
	}

	// Setup the request to Browserless content API
	reqURL := fmt.Sprintf("%s/content?token=%s", b.browserlessURL, b.token)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]interface{}{
		"url": url,
	}

	// Add optional parameters (only those that are supported)
	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		payload["waitFor"] = waitFor
	}

	// Convert payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	// Create the request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Apply client-side timeout
	clientCtx, cancel := context.WithTimeout(ctx, time.Duration(b.getTimeout(args))*time.Second)
	defer cancel()
	req = req.WithContext(clientCtx)

	// Send the request
	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("browserless error: %s, status: %d", string(body), resp.StatusCode)
	}

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"html":   string(body),
	}, nil
}

// screenshot takes a screenshot of a URL
func (b *BrowserTool) screenshot(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, errors.New("url must be a string")
	}

	// Setup the request to Browserless screenshot API
	reqURL := fmt.Sprintf("%s/screenshot?token=%s", b.browserlessURL, b.token)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]interface{}{
		"url": url,
	}

	// Add optional parameters (only those that are supported)
	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		payload["waitFor"] = waitFor
	}

	if fullPage, ok := args["fullPage"].(bool); ok {
		payload["options"] = map[string]interface{}{
			"fullPage": fullPage,
		}
	}

	// Convert payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	// Create the request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Apply client-side timeout
	clientCtx, cancel := context.WithTimeout(ctx, time.Duration(b.getTimeout(args))*time.Second)
	defer cancel()
	req = req.WithContext(clientCtx)

	// Send the request
	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("browserless error: %s, status: %d", string(body), resp.StatusCode)
	}

	// Read the response (binary data)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Convert to base64 for easier handling
	base64Screenshot := base64.StdEncoding.EncodeToString(body)

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"data":   "data:image/png;base64," + base64Screenshot,
	}, nil
}

// generatePDF generates a PDF of a URL
func (b *BrowserTool) generatePDF(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, errors.New("url must be a string")
	}

	// Setup the request to Browserless PDF API
	reqURL := fmt.Sprintf("%s/pdf?token=%s", b.browserlessURL, b.token)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]interface{}{
		"url": url,
	}

	// Add optional parameters (only those that are supported)
	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		payload["waitFor"] = waitFor
	}

	pdfOptions := map[string]interface{}{
		"printBackground": true,
	}

	if fullPage, ok := args["fullPage"].(bool); ok && fullPage {
		pdfOptions["preferCSSPageSize"] = true
	}

	payload["options"] = pdfOptions

	// Convert payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	// Create the request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Apply client-side timeout
	clientCtx, cancel := context.WithTimeout(ctx, time.Duration(b.getTimeout(args))*time.Second)
	defer cancel()
	req = req.WithContext(clientCtx)

	// Send the request
	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("browserless error: %s, status: %d", string(body), resp.StatusCode)
	}

	// Read the response (binary data)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Convert to base64 for easier handling
	base64PDF := base64.StdEncoding.EncodeToString(body)

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"data":   "data:application/pdf;base64," + base64PDF,
	}, nil
}

// extractContent extracts content from a webpage using a selector
func (b *BrowserTool) extractContent(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, errors.New("url must be a string")
	}

	selector, ok := args["selector"].(string)
	if !ok {
		return nil, errors.New("selector must be a string")
	}

	// Setup the request to Browserless function API
	reqURL := fmt.Sprintf("%s/function?token=%s", b.browserlessURL, b.token)

	// JavaScript to extract content - using ES Module format with export default
	extractScript := fmt.Sprintf(`
		export default async function ({ page }) {
			// Navigate to the page
			await page.goto("%s", { waitUntil: 'networkidle2' });
			
			// Wait for selector if specified
			const waitForSelector = "%s";
			if (waitForSelector) {
				try {
					await page.waitForSelector(waitForSelector, { timeout: 30000 });
				} catch (e) {
					// Continue even if timeout occurs
				}
			}
			
			// Extract content using the selector
			const selector = "%s";
			const results = await page.evaluate((sel) => {
				const elements = Array.from(document.querySelectorAll(sel));
				return elements.map(el => {
					return {
						text: el.innerText,
						html: el.innerHTML,
						attributes: Array.from(el.attributes).reduce((obj, attr) => {
							obj[attr.name] = attr.value;
							return obj;
						}, {})
					};
				});
			}, selector);
			
			// Return data in the format expected by Browserless
			return {
				data: {
					url: await page.url(),
					results: results
				},
				type: "application/json"
			};
		}
	`, url, getStringOr(args, "wait_for", ""), selector)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]interface{}{
		"code": extractScript,
	}

	// Convert payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	// Create the request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Apply client-side timeout
	clientCtx, cancel := context.WithTimeout(ctx, time.Duration(b.getTimeout(args))*time.Second)
	defer cancel()
	req = req.WithContext(clientCtx)

	// Send the request
	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("browserless error: %s, status: %d", string(body), resp.StatusCode)
	}

	// Read and parse the response
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return map[string]interface{}{
		"status":  "success",
		"url":     url,
		"content": result,
	}, nil
}

// executeScript executes a custom JavaScript in the browser
func (b *BrowserTool) executeScript(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	script, ok := args["script"].(string)
	if !ok {
		return nil, errors.New("script must be a string")
	}

	// Get the URL if provided
	targetURL := getStringOr(args, "url", "about:blank")

	// Setup the request to Browserless function API
	reqURL := fmt.Sprintf("%s/function?token=%s", b.browserlessURL, b.token)

	// Wrap the user's script in ES Module format with export default
	wrappedScript := fmt.Sprintf(`
		export default async function ({ page }) {
			// Navigate to the page if URL is provided
			await page.goto("%s", { waitUntil: 'networkidle2' });
			
			// Wait for selector if specified
			const waitForSelector = "%s";
			if (waitForSelector) {
				try {
					await page.waitForSelector(waitForSelector, { timeout: 30000 });
				} catch (e) {
					// Continue even if timeout occurs
				}
			}
			
			// Execute the user's script
			const result = await page.evaluate(() => {
				%s
			});
			
			// Return data in the format expected by Browserless
			return {
				data: result,
				type: "application/json"
			};
		}
	`, targetURL, getStringOr(args, "wait_for", ""), script)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]interface{}{
		"code": wrappedScript,
	}

	// Convert payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	// Create the request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Apply client-side timeout
	clientCtx, cancel := context.WithTimeout(ctx, time.Duration(b.getTimeout(args))*time.Second)
	defer cancel()
	req = req.WithContext(clientCtx)

	// Send the request
	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("browserless error: %s, status: %d", string(body), resp.StatusCode)
	}

	// Read and parse the response
	var result interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return map[string]interface{}{
		"status": "success",
		"result": result,
	}, nil
}

// getTimeout gets the timeout from args or uses the default
func (b *BrowserTool) getTimeout(args map[string]interface{}) int {
	if timeout, ok := args["timeout"].(float64); ok {
		return int(timeout)
	}
	return b.timeout
}

// getStringOr gets a string from args or returns the default
func getStringOr(args map[string]interface{}, key, defaultValue string) string {
	if value, ok := args[key].(string); ok {
		return value
	}
	return defaultValue
}
