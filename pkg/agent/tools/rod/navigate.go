package rod

import (
	"context"
	"fmt"
	"time"

	"github.com/go-rod/rod"
	"github.com/go-rod/rod/lib/launcher"
	"github.com/go-rod/rod/lib/proto"
	"github.com/go-rod/stealth"
)

// navigate navigates to a URL and returns the page HTML
func (t *Tool) navigate(ctx context.Context, args map[string]any) (any, error) {
	url, ok := args["url"].(string)

	if !ok {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("url must be a string"))
	}

	timeout := getTimeoutDuration(args, t.timeout)

	l := launcher.New().
		Headless(false).
		Set("disable-web-security", "").
		Set("disable-setuid-sandbox", "").
		Set("no-sandbox", "")

	debugURL, err := l.Launch()

	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to launch browser: %w", err))
	}

	browser := rod.New().
		ControlURL(debugURL).
		MustConnect()

	t.page, err = stealth.Page(browser)

	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to create stealth page: %w", err))
	}

	browser.MustIgnoreCertErrors(true)

	pageCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if err := t.page.Context(pageCtx).Navigate(url); err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to navigate to %s: %w", url, err))
	}

	t.page.WaitNavigation(
		proto.PageLifecycleEventNameNetworkAlmostIdle,
	)()

	if waitFor, ok := args["wait_for"].(string); ok && waitFor != "" {
		waitCtx, waitCancel := context.WithTimeout(ctx, 10*time.Second)
		defer waitCancel()

		if err := t.page.Context(waitCtx).MustElement(waitFor).WaitVisible(); err != nil {
			return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to wait for selector %s: %w", waitFor, err))
		}
	}

	html, err := t.page.HTML()

	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to get page HTML: %w", err))
	}

	return map[string]any{
		"status": "success",
		"url":    url,
		"html":   html,
	}, nil
}
