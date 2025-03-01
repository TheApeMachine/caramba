package rod

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"time"

	"github.com/go-rod/rod/lib/proto"
	"github.com/theapemachine/caramba/pkg/agent/util"
)

// generatePDF generates a PDF from a webpage
func (t *Tool) generatePDF(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, ok := args["url"].(string)
	if !ok || url == "" {
		return nil, fmt.Errorf("url must be a non-empty string")
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

	fullPage := getBoolOr(args, "full_page", false)
	format := getStringOr(args, "format", "Letter")

	pdfOpts := &proto.PagePrintToPDF{
		PrintBackground: true,
	}

	if format == "A4" {
		pdfOpts.PaperWidth = util.Ptr(8.27)
		pdfOpts.PaperHeight = util.Ptr(11.69)
	} else {
		pdfOpts.PaperWidth = util.Ptr(8.5)
		pdfOpts.PaperHeight = util.Ptr(11.0)
	}

	if fullPage {
		pdfOpts.PreferCSSPageSize = true
	}

	pdfReader, err := page.PDF(pdfOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to generate PDF: %w", err)
	}

	pdfData, err := io.ReadAll(pdfReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read PDF data: %w", err)
	}

	pdfBase64 := base64.StdEncoding.EncodeToString(pdfData)

	return map[string]interface{}{
		"status": "success",
		"url":    url,
		"data":   pdfBase64,
		"format": format,
	}, nil
}
