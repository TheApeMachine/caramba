package rod

import (
	"context"
	"fmt"
	"strings"
)

func (t *Tool) search(ctx context.Context, args map[string]any) (any, error) {
	query, ok := args["query"].(string)

	if !ok || query == "" {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("query must be a non-empty string"))
	}

	searchURL := fmt.Sprintf("https://duckduckgo.com/?q=%s&kp=-2&kl=us-en&kz=-1&kaf=1&k1=-1",
		strings.Replace(query, " ", "+", -1))

	_, err := t.navigate(ctx, map[string]interface{}{
		"url": searchURL,
	})

	if err != nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("search navigation failed: %w", err))
	}

	if t.page == nil {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("page not found"))
	}

	searchResults := t.page.MustEval(`() => {
		const currentDomain = window.location.hostname;
		return [...document.querySelectorAll('a')].map(a => {
			let score = 0;
			// Positional score
			if (!a.closest('header, footer, nav')) score += 3;
			if (a.closest('main, [role="main"], #content')) score += 2;
			
			// Content score
			if (a.textContent.trim().length > 20) score += 2;
			if (a.querySelector('img')) score += 1;
			
			// Link properties score
			if (!a.href.includes('#')) score += 1;
			if (!a.classList.contains('nav') && !a.id.includes('menu')) score += 1;
			
			return {
				href: a.href,
				text: a.textContent.trim(),
				score: score
			};
		}).filter(item => item.score >= 5).join("\n");
	}`).Arr()

	if len(searchResults) == 0 {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("no search results found"))
	}

	// Get HTML content for fallback
	html := t.page.MustEval(`() => document.documentElement.outerHTML`).String()

	// Process results into required format
	formattedResults := []map[string]string{
		{
			"summary": "Couldn't extract structured results, providing original HTML",
			"content": html[:1000] + "...",
		},
	}

	return map[string]any{
		"status":  "success",
		"url":     searchURL,
		"results": formattedResults,
		"count":   len(formattedResults),
	}, nil
}
