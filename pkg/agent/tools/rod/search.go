package rod

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/output"
)

// search performs a web search using the provided query
func (t *Tool) search(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return nil, fmt.Errorf("query must be a non-empty string")
	}

	output.Verbose(fmt.Sprintf("Searching for: %s", query))

	// Create an enhanced DuckDuckGo search URL
	searchURL := fmt.Sprintf("https://duckduckgo.com/?q=%s&kp=-2&kl=us-en&kz=-1&kaf=1&k1=-1",
		strings.Replace(query, " ", "+", -1))

	output.Debug(fmt.Sprintf("DuckDuckGo search URL: %s", searchURL))

	// Navigate and get the page content
	pageResult, err := t.navigate(ctx, map[string]interface{}{
		"url": searchURL,
	})
	if err != nil {
		return nil, fmt.Errorf("search navigation failed: %w", err)
	}

	// Process the search results to extract relevant information
	resultMap, ok := pageResult.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected result format")
	}

	html, ok := resultMap["html"].(string)
	if !ok {
		return nil, fmt.Errorf("html not found in result")
	}

	// Extract search results using custom extraction
	searchResults := t.extractSearchResults(html)

	// If we couldn't extract anything meaningful, fallback to a text summary
	if len(searchResults) == 0 {
		searchResults = []map[string]string{
			{
				"summary": "Couldn't extract structured results, providing original HTML",
				"content": html[:1000] + "...", // Truncate HTML to avoid overwhelming response
			},
		}
	}

	return map[string]interface{}{
		"status":  "success",
		"url":     searchURL,
		"results": searchResults,
		"count":   len(searchResults),
	}, nil
}

// extractSearchResults attempts to extract search results from the HTML
func (t *Tool) extractSearchResults(html string) []map[string]string {
	var results []map[string]string

	// Try to find result blocks in DuckDuckGo HTML
	resultBlocks := extractBetweenAll(html, `<div class="result`, `</div><!--result--`)

	// If no structured blocks found, try alternative patterns
	if len(resultBlocks) == 0 {
		resultBlocks = extractBetweenAll(html, `<div class="links_main`, `</div>`)
	}

	for _, block := range resultBlocks {
		result := make(map[string]string)

		// Extract the title
		title := extractBetween(block, `<a class="result__a" href="`, `</a>`)
		if title != "" {
			// Clean the title
			title = stripTags(title)
			title = strings.TrimSpace(title)
			result["title"] = title
		}

		// Extract the URL
		url := extractBetween(block, `<a class="result__a" href="`, `"`)
		if url != "" {
			result["url"] = url
		}

		// Extract the snippet
		snippet := extractBetween(block, `<a class="result__snippet"`, `</a>`)
		if snippet == "" {
			snippet = extractBetween(block, `<div class="result__snippet">`, `</div>`)
		}
		if snippet != "" {
			// Clean the snippet
			snippet = stripTags(snippet)
			snippet = strings.TrimSpace(snippet)
			result["snippet"] = snippet
		}

		// Only add if we have at least a title or snippet
		if result["title"] != "" || result["snippet"] != "" {
			results = append(results, result)
		}
	}

	return results
}
