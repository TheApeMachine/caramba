package browser

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/theapemachine/caramba/pkg/output"
)

// search performs a web search using the provided query
func (t *Tool) search(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return nil, errors.New("query must be a non-empty string")
	}

	output.Verbose(fmt.Sprintf("Searching for: %s", query))

	// Create an enhanced DuckDuckGo search URL with additional parameters:
	// - kp=-2: No safe search filtering
	// - kl=us-en: US English region
	// - kz=-1: No time filtering
	// - kaf=1: Show meanings when possible
	// - k1=-1: Advertisements disabled
	searchURL := fmt.Sprintf("https://duckduckgo.com/?q=%s&kp=-2&kl=us-en&kz=-1&kaf=1&k1=-1",
		strings.Replace(query, " ", "+", -1))

	output.Debug(fmt.Sprintf("DuckDuckGo search URL: %s", searchURL))

	// Create navigate args - only include parameters supported by Browserless API
	navigateArgs := map[string]interface{}{
		"url": searchURL,
	}

	// Use the navigate function to perform the search
	result, err := t.navigate(ctx, navigateArgs)
	if err != nil {
		return nil, err
	}

	// Clean up the HTML to extract only the relevant parts
	cleanedResult, err := t.cleanupSearchResults(result)
	if err != nil {
		return nil, fmt.Errorf("failed to clean up search results: %w", err)
	}

	return cleanedResult, nil
}

// cleanupSearchResults extracts the most relevant information from search results
// to avoid overflowing the context window with full HTML pages
func (t *Tool) cleanupSearchResults(result interface{}) (interface{}, error) {
	// Extract HTML from the result
	resultMap, ok := result.(map[string]interface{})
	if !ok {
		return nil, errors.New("unexpected result format")
	}

	html, ok := resultMap["html"].(string)
	if !ok {
		return nil, errors.New("html not found in result")
	}

	// Initialize cleaned output
	cleanedOutput := map[string]interface{}{
		"status": "success",
		"url":    resultMap["url"],
	}

	// Extract search results using a generic approach
	searchResults := extractSearchResults(html)

	// If we couldn't extract anything meaningful, fallback to a text summary
	if len(searchResults) == 0 {
		// Fallback: clean the HTML by removing scripts, styles, etc.
		cleanedText := cleanHTML(html)
		searchResults = []map[string]string{
			{
				"summary": "Couldn't extract structured results, providing cleaned text summary",
				"content": output.Summarize(cleanedText, 2000),
			},
		}
	}

	// Add cleaned results to output
	cleanedOutput["results"] = searchResults
	cleanedOutput["count"] = len(searchResults)

	// Include a shortened version of the HTML for debugging if needed
	cleanedOutput["html_sample"] = output.Summarize(html, 500)

	output.Debug(fmt.Sprintf("Extracted %d search results", len(searchResults)))
	return cleanedOutput, nil
}

// extractSearchResults attempts to extract search results from various search engines
func extractSearchResults(html string) []map[string]string {
	var results []map[string]string

	// Try to extract DuckDuckGo results
	results = extractDuckDuckGoResults(html)
	if len(results) > 0 {
		return results
	}

	// Could add other search engines here if needed

	return results
}

// extractDuckDuckGoResults extracts search results from DuckDuckGo HTML
func extractDuckDuckGoResults(html string) []map[string]string {
	var results []map[string]string

	// Try to find the main results container
	resultsHTML := extractBetween(html, `<div class="results`, `</div><!--results--`)
	if resultsHTML == "" {
		// Try alternative structure
		resultsHTML = extractBetween(html, `<div id="links" class="results`, `</div><!--links--`)
	}

	if resultsHTML == "" {
		// If we couldn't find the main container, use the entire HTML
		resultsHTML = html
	}

	// Extract individual result blocks
	resultBlocks := extractBetweenAll(resultsHTML, `<div class="result`, `</div><!--result--`)

	// If no structured blocks found, try alternative patterns
	if len(resultBlocks) == 0 {
		resultBlocks = extractBetweenAll(resultsHTML, `<div class="links_main`, `</div>`)
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

// cleanHTML removes non-content elements from HTML
func cleanHTML(html string) string {
	// Remove scripts
	html = removeBetweenAll(html, "<script", "</script>")

	// Remove styles
	html = removeBetweenAll(html, "<style", "</style>")

	// Remove HTML comments
	html = removeBetweenAll(html, "<!--", "-->")

	// Remove head section
	html = removeBetweenAll(html, "<head", "</head>")

	// Remove navigation
	html = removeBetweenAll(html, "<nav", "</nav>")

	// Remove footer
	html = removeBetweenAll(html, "<footer", "</footer>")

	// Strip all tags
	html = stripTags(html)

	// Normalize whitespace
	html = regexp.MustCompile(`\s+`).ReplaceAllString(html, " ")

	return strings.TrimSpace(html)
}

// extractBetween extracts text between startStr and endStr
func extractBetween(text, startStr, endStr string) string {
	startIdx := strings.Index(text, startStr)
	if startIdx == -1 {
		return ""
	}

	startIdx += len(startStr)
	endIdx := strings.Index(text[startIdx:], endStr)
	if endIdx == -1 {
		return ""
	}

	return text[startIdx : startIdx+endIdx]
}

// extractBetweenAll extracts all occurrences of text between startStr and endStr
func extractBetweenAll(text, startStr, endStr string) []string {
	var results []string

	for {
		startIdx := strings.Index(text, startStr)
		if startIdx == -1 {
			break
		}

		startIdx += len(startStr)
		endIdx := strings.Index(text[startIdx:], endStr)
		if endIdx == -1 {
			break
		}

		results = append(results, text[startIdx:startIdx+endIdx])
		text = text[startIdx+endIdx+len(endStr):]
	}

	return results
}

// removeBetweenAll removes all text between startStr and endStr, including the delimiters
func removeBetweenAll(text, startStr, endStr string) string {
	for {
		startIdx := strings.Index(text, startStr)
		if startIdx == -1 {
			break
		}

		endIdx := strings.Index(text[startIdx:], endStr)
		if endIdx == -1 {
			break
		}

		text = text[:startIdx] + text[startIdx+endIdx+len(endStr):]
	}

	return text
}

// stripTags removes all HTML tags from text
func stripTags(text string) string {
	return regexp.MustCompile(`<[^>]*>`).ReplaceAllString(text, "")
}
