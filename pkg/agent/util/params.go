package util

import (
	"github.com/charmbracelet/log"
	"github.com/pkoukk/tiktoken-go"
)

func Truncate(msgs map[string]string) map[string]string {
	// Always include first two messages (system prompt and user message)
	if len(msgs) < 2 {
		return msgs
	}

	maxTokens := 8000 - 500 // Reserve tokens for response
	totalTokens := 0
	var truncatedMessages map[string]string

	// Add first two messages
	truncatedMessages = map[string]string{"role": msgs["role"], "content": msgs["content"]}
	totalTokens += EstimateTokens(map[string]string{"role": msgs["role"], "content": msgs["content"]})

	// Start from the most recent message for the rest
	for i := len(msgs) - 1; i >= 2; i-- {
		msg := msgs["role"]
		messageTokens := EstimateTokens(map[string]string{"role": msg, "content": ""})
		if totalTokens+messageTokens <= maxTokens {
			truncatedMessages = map[string]string{"role": msg, "content": ""}
			totalTokens += messageTokens
		} else {
			break
		}
	}

	return truncatedMessages
}

func EstimateTokens(msg map[string]string) int {
	encoding, err := tiktoken.EncodingForModel("gpt-4o-mini")

	if err != nil {
		log.Error("Error getting encoding", "error", err)
		return 0
	}

	tokensPerMessage := 4 // As per OpenAI's token estimation guidelines

	numTokens := tokensPerMessage
	numTokens += len(encoding.Encode(msg["role"], nil, nil))
	numTokens += len(encoding.Encode(msg["content"], nil, nil))

	return numTokens
}
