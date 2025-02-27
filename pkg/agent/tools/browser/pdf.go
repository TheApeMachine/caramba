package browser

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/output"
)

// generatePDF generates a PDF from a webpage and returns the PDF data in base64 format
func (t *Tool) generatePDF(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok || url == "" {
		return nil, errors.New("url must be a non-empty string")
	}

	// Extract options
	fullPage := getBoolOr(args, "full_page", false)
	format := getStringOr(args, "format", "Letter")
	selector := getStringOr(args, "selector", "")
	waitFor := getStringOr(args, "wait_for", "")
	timeout := getIntOr(args, "timeout", 30000)

	output.Verbose(fmt.Sprintf("Generating PDF for: %s", url))

	// Create the request payload
	requestBody := map[string]interface{}{
		"url": url,
		"options": map[string]interface{}{
			"fullPage": fullPage,
			"format":   format,
			"timeout":  timeout,
		},
	}

	// Add optional wait_for selector if provided
	if waitFor != "" {
		requestBody["waitFor"] = map[string]interface{}{
			"selector": waitFor,
		}
	}

	// Add optional selector if provided (for capturing only a portion of the page)
	if selector != "" {
		requestBody["options"].(map[string]interface{})["selector"] = selector
	}

	// Marshal the request body
	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	// Create the request
	pdfEndpoint := fmt.Sprintf("%s/pdf", t.apiBaseURL)
	req, err := t.createRequestWithAuth("POST", pdfEndpoint, bodyBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to create PDF request: %w", err)
	}

	// Send the request
	resp, err := t.client.Do(req.WithContext(ctx))
	if err != nil {
		return nil, fmt.Errorf("failed to send PDF request: %w", err)
	}
	defer resp.Body.Close()

	// Check for error status
	if resp.StatusCode >= 400 {
		var errorBody map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&errorBody); err == nil {
			if message, ok := errorBody["message"].(string); ok {
				return nil, fmt.Errorf("PDF generation error (status %d): %s", resp.StatusCode, message)
			}
		}
		return nil, fmt.Errorf("PDF generation error (status %d)", resp.StatusCode)
	}

	// Read the PDF data
	pdfData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read PDF data: %w", err)
	}

	// Encode PDF data to base64
	pdfBase64 := base64.StdEncoding.EncodeToString(pdfData)

	// Return the PDF data and metadata
	result := map[string]interface{}{
		"status": "success",
		"url":    url,
		"data":   pdfBase64,
		"format": format,
	}

	output.Debug(fmt.Sprintf("Generated PDF for %s (%d bytes)", url, len(pdfData)))
	return result, nil
}
