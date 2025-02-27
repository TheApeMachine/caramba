package rod

import (
	"context"
	"fmt"
	"time"

	"github.com/go-rod/rod/lib/proto"
	"github.com/theapemachine/caramba/pkg/output"
	"github.com/ysmood/gson"
)

// executeScript executes a custom JavaScript in the browser
func (t *Tool) executeScript(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	script, ok := args["script"].(string)
	if !ok {
		return nil, fmt.Errorf("script must be a string")
	}

	output.Verbose(fmt.Sprintf("Executing script (%d characters)", len(script)))

	// Get the URL if provided
	targetURL := getStringOr(args, "url", "about:blank")

	// Get timeout from args or use default
	timeout := getTimeoutDuration(args, t.timeout)

	// Create new page
	page := t.browser.MustPage()
	defer page.Close()

	// Set timeout context
	pageCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Navigate to the URL if it's not about:blank
	if targetURL != "about:blank" {
		if err := page.Context(pageCtx).Navigate(targetURL); err != nil {
			return nil, fmt.Errorf("failed to navigate to %s: %w", targetURL, err)
		}

		// Wait for network to be idle
		if err := page.WaitNavigation(proto.PageLifecycleEventNameNetworkAlmostIdle); err != nil {
			output.Verbose(fmt.Sprintf("Warning: timeout waiting for network idle: %v", err))
		}
	}

	// Wait for selector if specified
	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		waitCtx, waitCancel := context.WithTimeout(ctx, 10*time.Second)
		defer waitCancel()

		if err := page.Context(waitCtx).MustElement(waitFor).WaitVisible(); err != nil {
			output.Verbose(fmt.Sprintf("Warning: timeout waiting for selector %s: %v", waitFor, err))
		}
	}

	// Execute the script
	result, err := page.Eval(script)
	if err != nil {
		return nil, fmt.Errorf("script execution failed: %w", err)
	}

	// Convert the result to a Go value
	var goResult interface{}
	if result != nil {
		goResult = gson.New(result.Value).String()
	}

	output.Debug("Script execution completed successfully")

	return map[string]interface{}{
		"status": "success",
		"result": goResult,
	}, nil
}
