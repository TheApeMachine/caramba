package browser

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/output"
)

// executeScript executes a custom JavaScript in the browser
func (t *Tool) executeScript(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	script, ok := args["script"].(string)
	if !ok {
		return nil, errors.New("script must be a string")
	}

	output.Verbose(fmt.Sprintf("Executing script (%d characters)", len(script)))

	// Get the URL if provided
	targetURL := getStringOr(args, "url", "about:blank")

	// Setup the request to Browserless function API
	reqURL := fmt.Sprintf("%s/function", t.apiBaseURL)

	// Wrap the user's script in ES Module format with export default
	wrappedScript := fmt.Sprintf(`
		export default async function ({ page }) {
			// Navigate to the page if URL is provided
			await page.goto("%s", { waitUntil: 'networkidle2' });
			
			// Wait for selector if specified
			const waitForSelector = "%s";
			if (waitForSelector) {
				try {
					await page.waitForSelector(waitForSelector, { timeout: 30000 });
				} catch (e) {
					// Continue even if timeout occurs
				}
			}
			
			// Execute the user's script
			const result = await page.evaluate(() => {
				%s
			});
			
			// Return data in the format expected by Browserless
			return {
				data: result,
				type: "application/json"
			};
		}
	`, targetURL, getStringOr(args, "wait_for", ""), script)

	// Prepare the request payload - keeping it minimal to avoid validation errors
	payload := map[string]interface{}{
		"code": wrappedScript,
	}

	// Convert payload to JSON
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	// Create the request
	req, err := t.createRequestWithAuth("POST", reqURL, payloadBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Apply client-side timeout
	timeout := getIntOr(args, "timeout", 30)
	clientCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()
	req = req.WithContext(clientCtx)

	// Send the request
	resp, err := t.sendRequest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	output.Debug("Script execution completed successfully")

	return map[string]interface{}{
		"status": "success",
		"result": resp,
	}, nil
}
