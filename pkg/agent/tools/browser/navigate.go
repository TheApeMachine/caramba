package browser

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/theapemachine/caramba/pkg/hub"
)

// navigate navigates to a URL and returns the page HTML
func (t *Tool) navigate(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, errors.New("url must be a string")
	}

	// Setup the request to Browserless content API
	reqURL := fmt.Sprintf("%s/content", t.apiBaseURL)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]interface{}{
		"url": url,
	}

	// Add optional parameters (only those that are supported)
	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		payload["waitFor"] = waitFor
		t.hub.Add(&hub.Event{
			Origin:  t.Name(),
			Topic:   hub.TopicTypeAgent,
			Type:    hub.EventTypeToolCall,
			Message: fmt.Sprintf("Waiting for selector: %s", waitFor),
		})
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

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to read response: %w", err))
	}

	t.hub.Add(&hub.Event{
		Origin:  t.Name(),
		Topic:   hub.TopicTypeAgent,
		Type:    hub.EventTypeToolCall,
		Message: fmt.Sprintf("Retrieved %d bytes from %s", len(body), url),
	})

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"html":   string(body),
	}, nil
}
