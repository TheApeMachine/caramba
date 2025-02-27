package rod

import (
	"context"
	"fmt"
	"time"

	"github.com/go-rod/rod/lib/proto"
	"github.com/theapemachine/caramba/pkg/output"
)

// navigate navigates to a URL and returns the page HTML
func (t *Tool) navigate(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, fmt.Errorf("url must be a string")
	}

	output.Verbose(fmt.Sprintf("Navigating to: %s", url))

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

	// Get page HTML
	html, err := page.HTML()
	if err != nil {
		return nil, fmt.Errorf("failed to get page HTML: %w", err)
	}

	output.Debug(fmt.Sprintf("Retrieved %d bytes from %s", len(html), url))

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"html":   html,
	}, nil
}
