package rod

import (
	"context"
	"fmt"
	"strings"
)

func (t *Tool) search(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query, ok := args["query"].(string)

	if !ok || query == "" {
		return nil, fmt.Errorf("query must be a non-empty string")
	}

	searchURL := fmt.Sprintf("https://duckduckgo.com/?q=%s&kp=-2&kl=us-en&kz=-1&kaf=1&k1=-1",
		strings.Replace(query, " ", "+", -1))

	pageResult, err := t.navigate(ctx, map[string]interface{}{
		"url": searchURL,
	})

	if err != nil {
		return nil, fmt.Errorf("search navigation failed: %w", err)
	}

	resultMap, ok := pageResult.(map[string]interface{})

	if !ok {
		return nil, fmt.Errorf("unexpected result format")
	}

	html, ok := resultMap["html"].(string)

	if !ok {
		return nil, fmt.Errorf("html not found in result")
	}

	searchResults := t.extractSearchResults(html)

	if len(searchResults) == 0 {
		searchResults = []map[string]string{
			{
				"summary": "Couldn't extract structured results, providing original HTML",
				"content": html[:1000] + "...",
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

func (t *Tool) extractSearchResults(html string) []map[string]string {
	var results []map[string]string

	resultBlocks := extractBetweenAll(html, `<div class="result`, `</div><!--result--`)

	if len(resultBlocks) == 0 {
		resultBlocks = extractBetweenAll(html, `<div class="links_main`, `</div>`)
	}

	for _, block := range resultBlocks {
		result := make(map[string]string)

		title := extractBetween(block, `<a class="result__a" href="`, `</a>`)

		if title != "" {
			title = stripTags(title)
			title = strings.TrimSpace(title)
			result["title"] = title
		}

		url := extractBetween(block, `<a class="result__a" href="`, `"`)

		if url != "" {
			result["url"] = url
		}

		snippet := extractBetween(block, `<a class="result__snippet"`, `</a>`)

		if snippet == "" {
			snippet = extractBetween(block, `<div class="result__snippet">`, `</div>`)
		}

		if snippet != "" {
			snippet = stripTags(snippet)
			snippet = strings.TrimSpace(snippet)
			result["snippet"] = snippet
		}

		if result["title"] != "" || result["snippet"] != "" {
			results = append(results, result)
		}
	}

	return results
}
