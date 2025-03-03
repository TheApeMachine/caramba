package rod

import (
	"context"
	"encoding/base64"
	"fmt"
	"time"

	"github.com/go-rod/rod/lib/proto"
)

// screenshot takes a screenshot of a URL
func (t *Tool) screenshot(ctx context.Context, args map[string]any) (any, error) {
	url, ok := args["url"].(string)

	if !ok {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("url must be a string"))
	}

	timeout := getTimeoutDuration(args, t.timeout)

	page := t.browser.MustPage()
	defer page.Close()

	pageCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if err := page.Context(pageCtx).Navigate(url); err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to navigate to %s: %w", url, err))
	}

	page.WaitNavigation(
		proto.PageLifecycleEventNameNetworkAlmostIdle,
	)()

	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		waitCtx, waitCancel := context.WithTimeout(ctx, 10*time.Second)
		defer waitCancel()

		if err := page.Context(waitCtx).MustElement(waitFor).WaitVisible(); err != nil {
			return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to wait for selector %s: %w", waitFor, err))
		}
	}

	fullPage := false

	if val, ok := args["full_page"].(bool); ok {
		fullPage = val
	}

	var img []byte
	var err error

	if fullPage {
		img, err = page.Screenshot(true, nil)
	} else {
		img, err = page.Screenshot(false, nil)
	}

	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to take screenshot: %w", err))
	}

	base64Screenshot := base64.StdEncoding.EncodeToString(img)

	return map[string]any{
		"status": "success",
		"url":    url,
		"data":   "data:image/png;base64," + base64Screenshot,
	}, nil
}
