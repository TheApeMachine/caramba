package rod

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"time"

	"github.com/go-rod/rod/lib/proto"
	"github.com/theapemachine/caramba/pkg/agent/util"
	"github.com/theapemachine/caramba/pkg/output"
)

// generatePDF generates a PDF from a webpage
func (t *Tool) generatePDF(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok || url == "" {
		return nil, fmt.Errorf("url must be a non-empty string")
	}

	output.Verbose(fmt.Sprintf("Generating PDF for: %s", url))

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

	// PDF options
	fullPage := getBoolOr(args, "full_page", false)
	format := getStringOr(args, "format", "Letter")

	// Create PDF options
	pdfOpts := &proto.PagePrintToPDF{
		PrintBackground: true,
	}

	// Adjust format if needed
	if format == "A4" {
		// A4 format
		pdfOpts.PaperWidth = util.Ptr(8.27)   // A4 width in inches
		pdfOpts.PaperHeight = util.Ptr(11.69) // A4 height in inches
	} else {
		// Default Letter format
		pdfOpts.PaperWidth = util.Ptr(8.5)   // Letter width in inches
		pdfOpts.PaperHeight = util.Ptr(11.0) // Letter height in inches
	}

	// Set to print the entire page if requested
	if fullPage {
		pdfOpts.PreferCSSPageSize = true
	}

	// Generate the PDF
	pdfReader, err := page.PDF(pdfOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to generate PDF: %w", err)
	}

	// Read the PDF data
	pdfData, err := io.ReadAll(pdfReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read PDF data: %w", err)
	}

	// Encode PDF data to base64
	pdfBase64 := base64.StdEncoding.EncodeToString(pdfData)

	output.Debug(fmt.Sprintf("Generated PDF for %s (%d bytes)", url, len(pdfData)))

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"data":   pdfBase64,
		"format": format,
	}, nil
}
