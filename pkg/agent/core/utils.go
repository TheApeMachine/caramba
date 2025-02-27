package core

import (
	"encoding/json"
	"strings"

	"github.com/fatih/color"
	"github.com/theapemachine/caramba/pkg/output"
)

// estimateTokens is a rough approximation for counting tokens from text.
func estimateTokens(text string) int {
	words := strings.Fields(text)
	return len(words) * 4 / 3 // ~4/3 tokens per word as a naive guess
}

// isLikelyJSON checks if a string might be JSON based on a naive prefix check.
func isLikelyJSON(content string) bool {
	trimmed := strings.TrimSpace(content)
	return strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")
}

// maybeExtractContentField tries to parse JSON and extract top-level "content" field if found.
func maybeExtractContentField(content string) string {
	var jsonData map[string]interface{}
	if err := json.Unmarshal([]byte(content), &jsonData); err == nil {
		if cVal, ok := jsonData["content"]; ok && cVal != nil {
			if strContent, ok := cVal.(string); ok {
				return strContent
			}
		}
	}
	return content
}

// summarizeToolCallArgs returns a short textual summary for tool call args.
func summarizeToolCallArgs(toolCall ToolCall) string {
	argsJSON, _ := json.Marshal(toolCall.Args)
	return output.Summarize(string(argsJSON), 50)
}

// formatToolResult prints a nice heading and JSON for a tool result.
func formatToolResult(toolName string, result interface{}) string {
	var resultOutput string
	if resultStr, ok := result.(string); ok {
		resultOutput = resultStr
	} else {
		prettyResult, _ := json.MarshalIndent(result, "", "  ")
		resultOutput = string(prettyResult)
	}
	return resultOutput
}

// formatStreamedContent applies basic formatting to streamed content for demonstration.
func formatStreamedContent(content string) string {
	// For fun, we color certain markdown patterns:
	trimmed := strings.TrimSpace(content)

	switch {
	case strings.HasPrefix(trimmed, "# "):
		// Main header
		return color.New(color.FgHiCyan, color.Bold).Sprint(content)
	case strings.HasPrefix(trimmed, "## "):
		// Secondary header
		return color.New(color.FgCyan, color.Bold).Sprint(content)
	case strings.HasPrefix(trimmed, "### "):
		// Tertiary header
		return color.New(color.FgBlue, color.Bold).Sprint(content)
	case strings.HasPrefix(trimmed, "- "), strings.HasPrefix(trimmed, "* "):
		// List item
		return color.New(color.FgGreen).Sprint(content)
	case strings.HasPrefix(trimmed, "> "):
		// Blockquote
		return color.New(color.FgYellow).Sprint(content)
	case strings.HasPrefix(trimmed, "```"), strings.HasPrefix(trimmed, "`"):
		// Code
		return color.New(color.FgHiMagenta).Sprint(content)
	}
	// Default
	return content
}
