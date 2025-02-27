package rod

import (
	"context"
	"fmt"
	"time"

	"github.com/go-rod/rod/lib/proto"
	"github.com/theapemachine/caramba/pkg/output"
)

// extractContent extracts content from a webpage using a CSS selector
func (t *Tool) extractContent(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok || url == "" {
		return nil, fmt.Errorf("url must be a non-empty string")
	}

	selector, ok := args["selector"].(string)
	if !ok || selector == "" {
		return nil, fmt.Errorf("selector must be a non-empty string")
	}

	output.Verbose(fmt.Sprintf("Extracting content from %s using selector: %s", url, selector))

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
	waitFor := getStringOr(args, "wait_for", "")
	if waitFor != "" {
		waitCtx, waitCancel := context.WithTimeout(ctx, 10*time.Second)
		defer waitCancel()

		if err := page.Context(waitCtx).MustElement(waitFor).WaitVisible(); err != nil {
			output.Verbose(fmt.Sprintf("Warning: timeout waiting for selector %s: %v", waitFor, err))
		}
	}

	// Check what data to retrieve
	returnHTML := getBoolOr(args, "html", true)
	returnText := getBoolOr(args, "text", true)
	attribute := getStringOr(args, "attribute", "")

	// Find all elements matching the selector
	elements, err := page.Elements(selector)
	if err != nil {
		return nil, fmt.Errorf("failed to find elements with selector '%s': %w", selector, err)
	}

	// Extract data from each element
	var results []map[string]interface{}
	for i, element := range elements {
		result := map[string]interface{}{
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

	output.Debug(fmt.Sprintf("Extracted %d elements from %s using selector: %s", len(results), url, selector))

	return map[string]interface{}{
		"status":   "success",
		"url":      url,
		"selector": selector,
		"content": map[string]interface{}{
			"count":   len(results),
			"results": results,
		},
	}, nil
}
