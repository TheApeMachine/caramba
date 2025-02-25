package examples

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/memory"
	"github.com/theapemachine/caramba/pkg/agent/tools"
	"github.com/theapemachine/errnie"
)

// BrowserExample demonstrates how to use the browser tool with the default provider
func BrowserExample(apiKey, url string) error {
	// Set default URL if none provided
	if url == "" || url == "artificial intelligence" {
		url = "https://news.ycombinator.com"
	}

	// Set environment variables to force QDrant to use REST API
	os.Setenv("QDRANT_URL", "http://localhost:6333")
	os.Setenv("QDRANT_USE_REST", "true")

	// Create an embedding provider
	embeddingProvider := memory.NewOpenAIEmbeddingProvider(apiKey, "text-embedding-3-large")

	// Create base memory store
	baseMemory := memory.NewInMemoryStore()

	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Create unified memory with default options
	unifiedMemory, err := memory.NewUnifiedMemory(baseMemory, embeddingProvider, memory.DefaultUnifiedMemoryOptions())
	if err != nil {
		fmt.Printf("Warning: Failed to initialize unified memory: %v\n", err)
		fmt.Println("Continuing with basic in-memory store only...")
	}

	// Create tools
	browserTool := tools.NewBrowserTool("http://localhost:3000", "6R0W53R135510") // Using the token from docker-compose

	// Create an LLM provider with tools
	llmProvider := llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")

	// Create an agent
	agent := core.NewAgentBuilder("BrowserAgent").
		WithLLM(llmProvider).
		WithTool(browserTool).
		WithMemory(unifiedMemory).
		Build()

	return runBrowserExample(ctx, agent, url)
}

// runBrowserExample runs the browser example with the provided agent
func runBrowserExample(ctx context.Context, agent core.Agent, url string) error {
	fmt.Printf("Starting browser example with URL: %s\n", url)
	fmt.Println("1. Navigating to the URL and getting content...")

	// Step 1: Navigate to the URL and analyze the content
	response, err := agent.Execute(ctx, fmt.Sprintf(`
		You are a web browsing assistant that can interact with web pages.
		Please perform the following tasks:
		1. Navigate to %s 
		2. Extract the main title of the page
		3. Count how many links are on the page
		4. Summarize what the page is about based on its content

		Use the browser tool with the appropriate actions to complete these tasks.
		Present your findings in a structured format.
	`, url))
	if err != nil {
		errnie.Error(err)
		return err
	}
	fmt.Println("Navigation and analysis complete. Raw response:")
	fmt.Println(response)

	// Process and execute tool calls
	browserTool := getBrowserToolFromAgent(agent)
	toolCalls := parseToolCalls(response)
	if len(toolCalls) > 0 {
		fmt.Println("\nExecuting navigation tool calls...")
		for _, toolCall := range toolCalls {
			result, err := executeToolCall(ctx, browserTool, toolCall)
			if err != nil {
				fmt.Printf("Error executing tool call %s: %v\n", toolCall.Name, err)
				continue
			}
			fmt.Printf("\n--- %s Result ---\n", toolCall.Name)
			fmt.Println(formatToolResult(result))
			fmt.Println("-------------------")
		}
	}

	fmt.Println("\n--------------------------------------")

	// Step 2: Take a screenshot
	fmt.Println("2. Taking a screenshot...")
	screenshotResponse, err := agent.Execute(ctx, fmt.Sprintf(`
		Take a screenshot of the current page at %s. 
		Use the browser tool with the "screenshot" action and set fullPage to true.
		Return a brief message confirming the screenshot was taken.
	`, url))
	if err != nil {
		errnie.Error(err)
		return err
	}
	fmt.Println("Screenshot request complete. Raw response:")
	fmt.Println(screenshotResponse)

	// Process and execute screenshot tool call
	screenshotCalls := parseToolCalls(screenshotResponse)
	if len(screenshotCalls) > 0 {
		fmt.Println("\nTaking screenshot...")
		for _, toolCall := range screenshotCalls {
			if toolCall.Args["action"] == "screenshot" {
				result, err := executeToolCall(ctx, browserTool, toolCall)
				if err != nil {
					fmt.Printf("Error taking screenshot: %v\n", err)
					continue
				}

				// Save the screenshot
				screenshotPath := "screenshot.png"
				if err := saveScreenshotFromResult(result, screenshotPath); err != nil {
					fmt.Printf("Failed to save screenshot: %v\n", err)
				} else {
					absPath, _ := filepath.Abs(screenshotPath)
					fmt.Printf("Screenshot saved to: %s\n", absPath)
				}
			}
		}
	}

	fmt.Println("\n--------------------------------------")

	// Step 3: Extract specific information
	fmt.Println("3. Extracting specific information...")
	extractionResponse, err := agent.Execute(ctx, fmt.Sprintf(`
		Using the browser tool, please extract the following information from %s:
		
		1. All headline titles (h1, h2, h3 tags)
		2. The most upvoted or popular content (if available)
		3. Any timestamps or dates visible on the page
		
		Use the "extract" action with appropriate CSS selectors to target these elements.
		Present the information in a clear, organized format.
	`, url))
	if err != nil {
		errnie.Error(err)
		return err
	}
	fmt.Println("Information extraction complete. Raw response:")
	fmt.Println(extractionResponse)

	// Process and execute extraction tool calls
	extractionCalls := parseToolCalls(extractionResponse)
	if len(extractionCalls) > 0 {
		fmt.Println("\nExecuting extraction tool calls...")
		for _, toolCall := range extractionCalls {
			if toolCall.Args["action"] == "extract" {
				selector, _ := toolCall.Args["selector"].(string)
				fmt.Printf("\n--- Extraction Results for selector: %s ---\n", selector)

				result, err := executeToolCall(ctx, browserTool, toolCall)
				if err != nil {
					fmt.Printf("Error extracting content: %v\n", err)
					continue
				}

				fmt.Println(formatToolResult(result))
				fmt.Println("-------------------")
			}
		}
	}

	fmt.Println("\n--------------------------------------")

	// Step 4: Execute some custom JavaScript
	fmt.Println("4. Running custom JavaScript...")
	jsResponse, err := agent.Execute(ctx, `
		Using the browser tool with the "execute" action, run a JavaScript snippet that:
		
		1. Counts the number of links on the page
		2. Counts the number of images
		3. Measures the page's main content area (width and height)
		4. Counts the number of interactive elements (buttons, forms, etc.)
		
		Return this information as a structured object. Format the results in a readable way.
	`)
	if err != nil {
		errnie.Error(err)
		return err
	}
	fmt.Println("JavaScript execution complete. Raw response:")
	fmt.Println(jsResponse)

	// Process and execute JavaScript tool calls
	jsCalls := parseToolCalls(jsResponse)
	if len(jsCalls) > 0 {
		fmt.Println("\nExecuting JavaScript...")
		for _, toolCall := range jsCalls {
			if toolCall.Args["action"] == "execute" {
				result, err := executeToolCall(ctx, browserTool, toolCall)
				if err != nil {
					fmt.Printf("Error executing JavaScript: %v\n", err)
					continue
				}

				fmt.Println("\n--- JavaScript Execution Results ---")
				fmt.Println(formatToolResult(result))
				fmt.Println("-------------------")
			}
		}
	}

	fmt.Println("\n--------------------------------------")

	fmt.Println("Browser example completed successfully!")
	return nil
}

// getBrowserToolFromAgent retrieves the browser tool from an agent
func getBrowserToolFromAgent(agent core.Agent) core.Tool {
	// This is a simplified approach - in a real implementation, you'd want to properly extract the tool
	// For now, we'll create a new instance which should be functionally identical
	return tools.NewBrowserTool("http://localhost:3000", "6R0W53R135510")
}

// parseToolCalls parses tool calls from a JSON string
func parseToolCalls(response string) []core.ToolCall {
	// Try to parse the response as a JSON array of tool calls
	var toolCalls []core.ToolCall
	if err := json.Unmarshal([]byte(response), &toolCalls); err != nil {
		// If not an array, try parsing as a single tool call
		var singleToolCall core.ToolCall
		if err := json.Unmarshal([]byte(response), &singleToolCall); err == nil && singleToolCall.Name != "" {
			return []core.ToolCall{singleToolCall}
		}

		// If still failing, the response might not be a tool call at all
		return []core.ToolCall{}
	}

	return toolCalls
}

// executeToolCall executes a tool call and returns the result
func executeToolCall(ctx context.Context, tool core.Tool, toolCall core.ToolCall) (interface{}, error) {
	if tool.Name() != toolCall.Name {
		return nil, fmt.Errorf("tool name mismatch: expected %s, got %s", tool.Name(), toolCall.Name)
	}

	return tool.Execute(ctx, toolCall.Args)
}

// formatToolResult formats a tool result for display
func formatToolResult(result interface{}) string {
	// If the result is a string, return it directly
	if str, ok := result.(string); ok {
		return str
	}

	// Otherwise, marshal to JSON with indentation
	jsonBytes, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Sprintf("Error formatting result: %v", err)
	}

	return string(jsonBytes)
}

// saveScreenshotFromResult saves a screenshot from the tool execution result
func saveScreenshotFromResult(result interface{}, filename string) error {
	// Convert result to a string representation if it's not already
	var dataStr string

	switch v := result.(type) {
	case string:
		dataStr = v
	case map[string]interface{}:
		// For browser tool, screenshot result is often in "data" field
		if data, ok := v["data"].(string); ok {
			dataStr = data
		} else if jsonData, err := json.Marshal(v); err == nil {
			dataStr = string(jsonData)
		}
	default:
		if jsonData, err := json.Marshal(result); err == nil {
			dataStr = string(jsonData)
		}
	}

	// Try to find and extract base64 content
	base64Data := extractBase64FromString(dataStr)
	if base64Data == "" {
		return fmt.Errorf("no base64 image data found")
	}

	// Decode and save the image
	imageData, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		return fmt.Errorf("failed to decode base64 image: %v", err)
	}

	return os.WriteFile(filename, imageData, 0644)
}

// extractBase64FromString extracts base64 image data from a string
func extractBase64FromString(s string) string {
	// Look for common base64 image patterns
	base64Prefixes := []string{
		"data:image/png;base64,",
		"base64,",
	}

	for _, prefix := range base64Prefixes {
		if idx := strings.Index(s, prefix); idx >= 0 {
			startIdx := idx + len(prefix)
			endIdx := len(s)

			// Find the end of the base64 data
			for i := startIdx; i < len(s); i++ {
				if s[i] == '"' || s[i] == '\'' || s[i] == '`' || s[i] == '}' {
					endIdx = i
					break
				}
			}

			if startIdx < endIdx {
				return s[startIdx:endIdx]
			}
		}
	}

	// Fall back to looking for a very long base64-looking string
	// (This is a heuristic and not perfect)
	lines := strings.Split(s, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if len(trimmed) > 100 && isLikelyBase64(trimmed) {
			return trimmed
		}
	}

	return ""
}

// isLikelyBase64 checks if a string is likely to be base64 encoded
func isLikelyBase64(s string) bool {
	// Check if string only contains valid base64 characters
	for _, c := range s {
		if !(('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || ('0' <= c && c <= '9') || c == '+' || c == '/' || c == '=') {
			return false
		}
	}

	// Reasonable length heuristic for images
	return len(s) > 100
}

// TestMultiProviderBrowser runs the browser example with multiple LLM providers
func TestMultiProviderBrowser(openAIKey, anthropicKey, url string) error {
	if url == "" {
		url = "https://news.ycombinator.com"
	}

	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Create the browser tool
	browserTool := tools.NewBrowserTool("http://localhost:3000", "6R0W53R135510")

	// Test task to run for both providers
	testTask := fmt.Sprintf(`
		You are a web browser assistant. Navigate to %s and tell me:
		1. The page title
		2. How many links are on the page
		3. A brief summary of what you see

		Use the browser tool with the "navigate" action first, then use other actions as needed.
		Present your findings in a clear, concise format.
	`, url)

	// Test with OpenAI
	if openAIKey != "" {
		fmt.Println("\n=== Testing with OpenAI Provider ===\n")

		// Try to set up memory for OpenAI agent
		openaiAgentBuilder := core.NewAgentBuilder("OpenAIBrowserAgent").
			WithLLM(llm.NewOpenAIProvider(openAIKey, "gpt-4o-mini")).
			WithTool(browserTool)

		// Attempt to add memory if possible
		if memoryEnabled := tryAddMemory(openaiAgentBuilder, openAIKey); memoryEnabled {
			fmt.Println("Memory features are enabled for OpenAI agent.")
		}

		// Build the agent
		openaiAgent := openaiAgentBuilder.Build()

		result, err := openaiAgent.Execute(ctx, testTask)
		if err != nil {
			errnie.Error(err)
			fmt.Println("OpenAI test failed with error:", err)
		} else {
			fmt.Println("OpenAI Raw Result:")
			fmt.Println(result)

			// Process and execute tool calls
			toolCalls := parseToolCalls(result)
			if len(toolCalls) > 0 {
				fmt.Println("\nExecuting OpenAI tool calls...")
				for _, toolCall := range toolCalls {
					result, err := executeToolCall(ctx, browserTool, toolCall)
					if err != nil {
						fmt.Printf("Error executing tool call %s: %v\n", toolCall.Name, err)
						continue
					}
					fmt.Printf("\n--- %s Result ---\n", toolCall.Name)
					fmt.Println(formatToolResult(result))
					fmt.Println("-------------------")
				}
			}
		}
	}

	// Test with Anthropic
	if anthropicKey != "" {
		fmt.Println("\n=== Testing with Anthropic Provider ===\n")

		// Set up Anthropic agent
		anthropicAgentBuilder := core.NewAgentBuilder("AnthropicBrowserAgent").
			WithLLM(llm.NewAnthropicProvider(anthropicKey, "claude-3-haiku-20240307")).
			WithTool(browserTool)

		// Attempt to add memory if possible - we need OpenAI key for embeddings
		if openAIKey != "" {
			if memoryEnabled := tryAddMemory(anthropicAgentBuilder, openAIKey); memoryEnabled {
				fmt.Println("Memory features are enabled for Anthropic agent.")
			}
		}

		// Build the agent
		anthropicAgent := anthropicAgentBuilder.Build()

		result, err := anthropicAgent.Execute(ctx, testTask)
		if err != nil {
			errnie.Error(err)
			fmt.Println("Anthropic test failed with error:", err)
		} else {
			fmt.Println("Anthropic Raw Result:")
			fmt.Println(result)

			// Process and execute tool calls
			toolCalls := parseToolCalls(result)
			if len(toolCalls) > 0 {
				fmt.Println("\nExecuting Anthropic tool calls...")
				for _, toolCall := range toolCalls {
					result, err := executeToolCall(ctx, browserTool, toolCall)
					if err != nil {
						fmt.Printf("Error executing tool call %s: %v\n", toolCall.Name, err)
						continue
					}
					fmt.Printf("\n--- %s Result ---\n", toolCall.Name)
					fmt.Println(formatToolResult(result))
					fmt.Println("-------------------")
				}
			}
		}
	}

	return nil
}

// tryAddMemory attempts to initialize and add memory to an agent builder
// returns true if memory was successfully added
func tryAddMemory(builder *core.AgentBuilder, apiKey string) bool {
	// Create base memory and embedding provider
	baseMemory := memory.NewInMemoryStore()
	embeddingProvider := memory.NewOpenAIEmbeddingProvider(apiKey, "text-embedding-3-large")

	// Create unified memory - the implementation handles connection errors gracefully
	unifiedMemory, err := memory.NewUnifiedMemory(baseMemory, embeddingProvider, memory.DefaultUnifiedMemoryOptions())

	if err != nil {
		fmt.Printf("Warning: Failed to initialize unified memory: %v\n", err)
		fmt.Println("Agent will proceed without memory augmentation.")
		return false
	}

	// Add memory to the agent builder
	builder.WithMemory(unifiedMemory)
	return true
}
