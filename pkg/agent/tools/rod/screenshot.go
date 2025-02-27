package rod

import (
	"context"
	"encoding/base64"
	"fmt"
	"time"

	"github.com/go-rod/rod/lib/proto"
	"github.com/theapemachine/caramba/pkg/output"
)

// screenshot takes a screenshot of a URL
func (t *Tool) screenshot(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, fmt.Errorf("url must be a string")
	}

	output.Verbose(fmt.Sprintf("Taking screenshot of: %s", url))

	// Get timeout from args or use default
	timeout := getTimeoutDuration(args, t.timeout)

	// Create new page
	page := t.browser.MustPage()
	defer page.Close()

	// Set timeout context
	pageCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Navigate to the URL
	if err := page.Context(pageCtx).Navigate(url); err != nil {
		return nil, fmt.Errorf("failed to navigate to %s: %w", url, err)
	}

	// Wait for network to be idle
	if err := page.WaitNavigation(proto.PageLifecycleEventNameNetworkAlmostIdle); err != nil {
		output.Verbose(fmt.Sprintf("Warning: timeout waiting for network idle: %v", err))
	}

	// Wait for selector if specified
	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		waitCtx, waitCancel := context.WithTimeout(ctx, 10*time.Second)
		defer waitCancel()

		if err := page.Context(waitCtx).MustElement(waitFor).WaitVisible(); err != nil {
			output.Verbose(fmt.Sprintf("Warning: timeout waiting for selector %s: %v", waitFor, err))
		}
	}

	// Determine if we should capture full page
	fullPage := false
	if val, ok := args["full_page"].(bool); ok {
		fullPage = val
		if fullPage {
			output.Verbose("Capturing full page screenshot")
		}
	}

	// Take the screenshot
	var img []byte
	var err error

	if fullPage {
		// For full page screenshot, we don't need to specify quality
		img, err = page.Screenshot(true, nil)
	} else {
		// For viewport only screenshot, we don't need to specify quality
		img, err = page.Screenshot(false, nil)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to take screenshot: %w", err)
	}

	output.Debug(fmt.Sprintf("Captured screenshot (%d bytes)", len(img)))

	// Convert to base64 for easier handling
	base64Screenshot := base64.StdEncoding.EncodeToString(img)

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"data":   "data:image/png;base64," + base64Screenshot,
	}, nil
}
