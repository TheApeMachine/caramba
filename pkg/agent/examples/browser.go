package examples

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/tools"
	"github.com/theapemachine/errnie"
)

// BrowserExample demonstrates how to use the browser tool
func BrowserExample(apiKey, url string) error {
	if url == "" {
		url = "https://news.ycombinator.com"
	}

	// Create an LLM provider
	llmProvider := llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")

	// Create tools
	browserTool := tools.NewBrowserTool("", "6R0W53R135510") // Using the token from docker-compose

	// Create an agent
	agent := core.NewAgentBuilder("BrowserAgent").
		WithLLM(llmProvider).
		WithTool(browserTool).
		Build()

	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	fmt.Printf("Starting browser example with URL: %s\n", url)
	fmt.Println("1. Navigating to the URL and getting content...")

	// Step 1: Navigate to the URL
	response, err := agent.Execute(ctx, fmt.Sprintf(`
		You are a web browsing assistant. Please navigate to %s and return a summary 
		of what you see on the page. Use the browser tool with the "navigate" action.
	`, url))
	if err != nil {
		errnie.Error(err)
		return err
	}
	fmt.Println("Navigation complete. Response:")
	fmt.Println(response)
	fmt.Println("\n--------------------------------------\n")

	// Step 2: Take a screenshot
	fmt.Println("2. Taking a screenshot...")
	screenshotResponse, err := agent.Execute(ctx, fmt.Sprintf(`
		Take a screenshot of %s. Use the browser tool with the "screenshot" action and fullPage set to true.
		Return only the JSON result from the tool call.
	`, url))
	if err != nil {
		errnie.Error(err)
		return err
	}

	// Process the screenshot (simple extraction from response)
	// In a real application, you would parse the JSON response properly
	base64Screenshot := extractBase64FromResponse(screenshotResponse)
	if base64Screenshot != "" {
		// Save the screenshot
		if err := saveScreenshot(base64Screenshot, "screenshot.png"); err != nil {
			errnie.Info("Failed to save screenshot: " + err.Error())
		} else {
			fmt.Println("Screenshot saved to screenshot.png")
		}
	}
	fmt.Println("\n--------------------------------------\n")

	// Step 3: Extract specific content
	fmt.Println("3. Extracting specific content from the page...")
	extractResponse, err := agent.Execute(ctx, fmt.Sprintf(`
		Extract all link titles from %s. Use the browser tool with the "extract" action and a CSS selector that targets the 
		main headline links or article titles. Return a summarized list of the top 5 titles.
	`, url))
	if err != nil {
		errnie.Error(err)
		return err
	}
	fmt.Println("Content extraction complete. Response:")
	fmt.Println(extractResponse)
	fmt.Println("\n--------------------------------------\n")

	// Step 4: Execute custom JavaScript
	fmt.Println("4. Executing custom JavaScript on the page...")
	scriptResponse, err := agent.Execute(ctx, fmt.Sprintf(`
		Execute a JavaScript snippet on %s that counts the number of links, images, and paragraphs on the page.
		Use the browser tool with the "execute" action. The script should return an object with counts for each element type.
	`, url))
	if err != nil {
		errnie.Error(err)
		return err
	}
	fmt.Println("JavaScript execution complete. Response:")
	fmt.Println(scriptResponse)

	fmt.Println("\nBrowser example completed successfully!")
	return nil
}

// Helper function to extract base64 data from agent response
// This is a very simplified extraction - in a real app, use proper JSON parsing
func extractBase64FromResponse(response string) string {
	// Look for a common pattern that might surround base64 data in the response
	const base64Prefix = "base64,"
	prefixIndex := 0

	// Basic extraction - this is oversimplified
	// In a real application, properly parse the JSON response
	if prefixIndex > 0 {
		startIndex := prefixIndex + len(base64Prefix)
		endIndex := len(response)
		for i := startIndex; i < len(response); i++ {
			if response[i] == '"' || response[i] == '\'' || response[i] == '`' {
				endIndex = i
				break
			}
		}

		if startIndex < endIndex {
			return response[startIndex:endIndex]
		}
	}

	// Fallback - scan for a long base64-looking string
	// This is not reliable and is only for demonstration purposes
	return ""
}

// Save a base64-encoded image to a file
func saveScreenshot(base64Image, filename string) error {
	imageData, err := base64.StdEncoding.DecodeString(base64Image)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, imageData, 0644)
}
