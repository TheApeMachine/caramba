package rod

import (
	"context"
	"fmt"
	"time"

	"github.com/go-rod/rod/lib/proto"
)

// navigate navigates to a URL and returns the page HTML
func (t *Tool) navigate(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)

	if !ok {
		return nil, fmt.Errorf("url must be a string")
	}

	timeout := getTimeoutDuration(args, t.timeout)

	page := t.browser.MustPage()
	defer page.Close()

	pageCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if err := page.Context(pageCtx).Navigate(url); err != nil {
		return nil, fmt.Errorf("failed to navigate to %s: %w", url, err)
	}

	page.WaitNavigation(
		proto.PageLifecycleEventNameNetworkAlmostIdle,
	)()

	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		waitCtx, waitCancel := context.WithTimeout(ctx, 10*time.Second)
		defer waitCancel()

		if err := page.Context(waitCtx).MustElement(waitFor).WaitVisible(); err != nil {
			return nil, fmt.Errorf("failed to wait for selector %s: %w", waitFor, err)
		}
	}

	html, err := page.HTML()

	if err != nil {
		return nil, fmt.Errorf("failed to get page HTML: %w", err)
	}

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"html":   html,
	}, nil
}
