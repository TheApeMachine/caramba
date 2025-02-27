package browser

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/theapemachine/caramba/pkg/output"
)

// screenshot takes a screenshot of a URL
func (t *Tool) screenshot(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, errors.New("url must be a string")
	}

	// Setup the request to Browserless screenshot API
	reqURL := fmt.Sprintf("%s/screenshot", t.apiBaseURL)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]interface{}{
		"url": url,
	}

	// Add optional parameters (only those that are supported)
	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		payload["waitFor"] = waitFor
		output.Verbose(fmt.Sprintf("Waiting for selector: %s", waitFor))
	}

	if fullPage, ok := args["fullPage"].(bool); ok {
		payload["options"] = map[string]interface{}{
			"fullPage": fullPage,
		}
		if fullPage {
			output.Verbose("Capturing full page screenshot")
		}
	}

	// Convert payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	// Create the request
	req, err := t.createRequestWithAuth("POST", reqURL, payloadBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Apply client-side timeout
	timeout := getIntOr(args, "timeout", 30)
	clientCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()
	req = req.WithContext(clientCtx)

	// Send the request
	resp, err := t.client.Do(req)
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

	output.Debug(fmt.Sprintf("Captured screenshot (%d bytes)", len(body)))

	// Convert to base64 for easier handling
	base64Screenshot := base64.StdEncoding.EncodeToString(body)

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"data":   "data:image/png;base64," + base64Screenshot,
	}, nil
}
