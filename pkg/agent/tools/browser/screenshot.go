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

	"github.com/theapemachine/caramba/pkg/hub"
)

// screenshot takes a screenshot of a URL
func (t *Tool) screenshot(ctx context.Context, args map[string]any) (any, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, errors.New("url must be a string")
	}

	// Setup the request to Browserless screenshot API
	reqURL := fmt.Sprintf("%s/screenshot", t.apiBaseURL)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]any{
		"url": url,
	}

	// Add optional parameters (only those that are supported)
	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		payload["waitFor"] = waitFor
	}

	if fullPage, ok := args["fullPage"].(bool); ok {
		payload["options"] = map[string]any{
			"fullPage": fullPage,
		}
		if fullPage {
			t.hub.Add(&hub.Event{
				Origin:  t.Name(),
				Topic:   hub.TopicTypeAgent,
				Type:    hub.EventTypeToolCall,
				Message: "Capturing full page screenshot",
			})
		}
	}

	// Convert payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to marshal request payload: %w", err))
	}

	// Create the request
	req, err := t.createRequestWithAuth("POST", reqURL, payloadBytes)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to create request: %w", err))
	}

	// Apply client-side timeout
	timeout := getIntOr(args, "timeout", 30)
	clientCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()
	req = req.WithContext(clientCtx)

	// Send the request
	resp, err := t.client.Do(req)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to send request: %w", err))
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, t.logger.Error(t.Name(), fmt.Errorf("browserless error: %s, status: %d", string(body), resp.StatusCode))
	}

	// Read the response (binary data)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to read response: %w", err))
	}

	t.hub.Add(&hub.Event{
		Origin:  t.Name(),
		Topic:   hub.TopicTypeAgent,
		Type:    hub.EventTypeToolCall,
		Message: fmt.Sprintf("Captured screenshot (%d bytes)", len(body)),
	})

	// Convert to base64 for easier handling
	base64Screenshot := base64.StdEncoding.EncodeToString(body)

	return map[string]any{
		"status": "success",
		"url":    url,
		"data":   "data:image/png;base64," + base64Screenshot,
	}, nil
}
