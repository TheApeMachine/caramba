package browser

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/hub"
)

// extractContent extracts content from a webpage using a CSS selector
func (t *Tool) extractContent(ctx context.Context, args map[string]any) (any, error) {
	url, ok := args["url"].(string)
	if !ok || url == "" {
		return nil, t.logger.Error(t.Name(), errors.New("url must be a non-empty string"))
	}

	selector, ok := args["selector"].(string)
	if !ok || selector == "" {
		return nil, t.logger.Error(t.Name(), errors.New("selector must be a non-empty string"))
	}

	// Extract options
	waitFor := getStringOr(args, "wait_for", "")
	timeout := getIntOr(args, "timeout", 30000)
	attribute := getStringOr(args, "attribute", "")
	returnHTML := getBoolOr(args, "html", true)
	returnText := getBoolOr(args, "text", true)

	// Construct a function to extract content
	extractScript := fmt.Sprintf(`
		const elements = document.querySelectorAll("%s");
		const results = [];
		
		elements.forEach((el, index) => {
			const result = {
				index: index,
				exists: true,
			};
			
			%s
			
			%s
			
			%s
			
			results.push(result);
		});
		
		return {
			count: results.length,
			results: results
		};
	`,
		escapeJSString(selector),
		getHTMLScript(returnHTML),
		getTextScript(returnText),
		getAttributeScript(attribute),
	)

	// Create the request payload for Browserless function API
	requestBody := map[string]interface{}{
		"code": fmt.Sprintf(`
			module.exports = async function browserlessFunction(context) {
				const { page } = context;
				await page.goto("%s", { waitUntil: "networkidle2", timeout: %d });
				%s
				const result = await page.evaluate(%s);
				return result;
			}
		`,
			escapeJSString(url),
			timeout,
			getWaitForScript(waitFor),
			extractScript,
		),
	}

	// Marshal the request body
	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to marshal request body: %w", err))
	}

	// Create the request
	functionEndpoint := fmt.Sprintf("%s/function", t.apiBaseURL)
	req, err := t.createRequestWithAuth("POST", functionEndpoint, bodyBytes)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to create extract request: %w", err))
	}

	// Send the request
	resp, err := t.sendRequest(ctx, req)
	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("failed to extract content: %w", err))
	}

	// Format and return the result
	result := map[string]any{
		"status":   "success",
		"url":      url,
		"selector": selector,
		"content":  resp,
	}

	// Log extraction results
	count := 0
	if data, ok := resp["data"].(map[string]any); ok {
		if c, ok := data["count"].(float64); ok {
			count = int(c)
		}
	}

	t.hub.Add(&hub.Event{
		Origin:  t.Name(),
		Topic:   hub.TopicTypeAgent,
		Type:    hub.EventTypeToolCall,
		Message: fmt.Sprintf("%d", count),
	})

	return result, nil
}

// Helper functions for building the extraction script

// escapeJSString escapes a string for JavaScript
func escapeJSString(s string) string {
	// Simple implementation - for production, use a proper JS string escaping library
	replacer := map[string]string{
		"\\": "\\\\",
		"\"": "\\\"",
		"\n": "\\n",
		"\r": "\\r",
		"\t": "\\t",
	}

	for char, replacement := range replacer {
		s = strings.Replace(s, char, replacement, -1)
	}
	return s
}

// getWaitForScript generates script to wait for an element if specified
func getWaitForScript(waitFor string) string {
	if waitFor == "" {
		return ""
	}
	return fmt.Sprintf(`
		try {
			await page.waitForSelector("%s", { timeout: 5000 });
		} catch (err) {
			console.warn("Wait for selector timed out: %s");
		}
	`, escapeJSString(waitFor), escapeJSString(waitFor))
}

// getHTMLScript generates script to extract HTML if required
func getHTMLScript(returnHTML bool) string {
	if !returnHTML {
		return ""
	}
	return `
		try {
			result.html = el.outerHTML;
		} catch (err) {
			result.html = null;
			result.htmlError = err.message;
		}
	`
}

// getTextScript generates script to extract text if required
func getTextScript(returnText bool) string {
	if !returnText {
		return ""
	}
	return `
		try {
			result.text = el.textContent.trim();
		} catch (err) {
			result.text = null;
			result.textError = err.message;
		}
	`
}

// getAttributeScript generates script to extract an attribute if specified
func getAttributeScript(attribute string) string {
	if attribute == "" {
		return ""
	}
	return fmt.Sprintf(`
		try {
			result.attribute = {
				name: "%s",
				value: el.getAttribute("%s")
			};
		} catch (err) {
			result.attribute = {
				name: "%s",
				value: null,
				error: err.message
			};
		}
	`, escapeJSString(attribute), escapeJSString(attribute), escapeJSString(attribute))
}
