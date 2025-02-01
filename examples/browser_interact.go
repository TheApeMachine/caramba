package examples

import (
	"encoding/json"
	"fmt"

	"github.com/theapemachine/caramba/tools"
)

func RunBrowserInteractExample() {
	browser := tools.NewBrowser()

	// Navigate to GitHub
	args := map[string]any{
		"url":      "https://github.com",
		"selector": "[name='q']", // Search input
		"action":   "click",
	}

	if _, err := browser.Run(args); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Type search query
	args = map[string]any{
		"hotkeys": "golang awesome",
	}

	if _, err := browser.Run(args); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Take screenshot
	args = map[string]any{
		"screenshot": true,
	}

	result, err := browser.Run(args)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	var browserResult tools.BrowserResult
	if err := json.Unmarshal([]byte(result), &browserResult); err != nil {
		fmt.Printf("Error parsing result: %v\n", err)
		return
	}

	fmt.Printf("Screenshot taken, size: %d bytes\n", len(browserResult.Screenshot))

	browser.Close()
}
