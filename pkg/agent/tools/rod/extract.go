package rod

import (
	"context"
	"fmt"
	"time"

	"github.com/go-rod/rod/lib/proto"
)

// extractContent extracts content from a webpage using a CSS selector
func (t *Tool) extractContent(ctx context.Context, args map[string]any) (any, error) {
	url, ok := args["url"].(string)

	if !ok || url == "" {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("url must be a non-empty string"))
	}

	selector, ok := args["selector"].(string)

	if !ok || selector == "" {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("selector must be a non-empty string"))
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

	// Wait for selector if specified
	waitFor := getStringOr(args, "wait_for", "")
	if waitFor != "" {
		waitCtx, waitCancel := context.WithTimeout(ctx, 10*time.Second)
		defer waitCancel()

		if err := page.Context(waitCtx).MustElement(waitFor).WaitVisible(); err != nil {
			t.logger.Error(t.Name(), fmt.Errorf("timeout waiting for selector %s: %w", waitFor, err))
		}
	}

	returnHTML := getBoolOr(args, "html", true)
	returnText := getBoolOr(args, "text", true)
	attribute := getStringOr(args, "attribute", "")

	elements, err := page.Elements(selector)

	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to find elements with selector '%s': %w", selector, err))
	}

	var results []map[string]any

	for i, element := range elements {
		result := map[string]any{
			"index":  i,
			"exists": true,
		}

		if returnHTML {
			html, err := element.HTML()
			if err == nil {
				result["html"] = html
			} else {
				result["htmlError"] = err.Error()
			}
		}

		if returnText {
			text, err := element.Text()
			if err == nil {
				result["text"] = text
			} else {
				result["textError"] = err.Error()
			}
		}

		if attribute != "" {
			attrValue, err := element.Attribute(attribute)
			if err == nil && attrValue != nil {
				result["attribute"] = map[string]string{
					"name":  attribute,
					"value": *attrValue,
				}
			} else {
				result["attributeError"] = "Attribute not found or error"
			}
		}

		results = append(results, result)
	}

	return map[string]any{
		"status":   "success",
		"url":      url,
		"selector": selector,
		"content": map[string]any{
			"count":   len(results),
			"results": results,
		},
	}, nil
}
