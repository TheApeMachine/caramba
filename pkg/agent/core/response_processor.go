package core

import (
	"encoding/json"
	"strings"
)

// ResponseProcessor handles the processing of LLM responses.
type ResponseProcessor struct{}

// NewResponseProcessor creates a new ResponseProcessor.
func NewResponseProcessor() *ResponseProcessor {
	return &ResponseProcessor{}
}

// ProcessChunkContent processes content from a streaming chunk.
func (rp *ResponseProcessor) ProcessChunkContent(content string) string {
	if isLikelyJSON(content) {
		// Attempt to extract "content" field if it's JSON
		content = maybeExtractContentField(content)
	}
	return content
}

// FormatStreamedContent applies formatting to streamed content.
func (rp *ResponseProcessor) FormatStreamedContent(content string) string {
	return formatStreamedContent(content)
}

// ExtractToolCalls tries to parse a response for tool calls.
func (rp *ResponseProcessor) ExtractToolCalls(response string) []ToolCall {
	var toolCalls []ToolCall
	if err := json.Unmarshal([]byte(response), &toolCalls); err == nil && len(toolCalls) > 0 {
		return toolCalls
	}
	// Try single tool call
	var singleToolCall ToolCall
	if err := json.Unmarshal([]byte(response), &singleToolCall); err == nil && singleToolCall.Name != "" {
		return []ToolCall{singleToolCall}
	}
	// Fallback: no recognized calls
	return []ToolCall{}
}

// SummarizeToolCallArgs returns a short textual summary for tool call args.
func (rp *ResponseProcessor) SummarizeToolCallArgs(toolCall ToolCall) string {
	argsJSON, _ := json.Marshal(toolCall.Args)
	return summarizeString(string(argsJSON), 50)
}

// FormatToolResult formats a tool result for display.
func (rp *ResponseProcessor) FormatToolResult(toolName string, result interface{}) string {
	var resultOutput string
	if resultStr, ok := result.(string); ok {
		resultOutput = resultStr
	} else {
		prettyResult, _ := json.MarshalIndent(result, "", "  ")
		resultOutput = string(prettyResult)
	}
	return resultOutput
}

// summarizeString returns a truncated version of a string with ellipsis if needed.
func summarizeString(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen-3] + "..."
}

// EstimateTokens provides a rough approximation for counting tokens from text.
func (rp *ResponseProcessor) EstimateTokens(text string) int {
	words := strings.Fields(text)
	return len(words) * 4 / 3 // ~4/3 tokens per word as a naive guess
}
