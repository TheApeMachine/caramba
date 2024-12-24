package ai

import (
	"github.com/charmbracelet/log"
	"github.com/pkoukk/tiktoken-go"
	"github.com/theapemachine/caramba/provider"
)

type Buffer struct {
	messages         []provider.Message
	maxContextTokens int
}

func NewBuffer() *Buffer {
	return &Buffer{
		messages:         make([]provider.Message, 0),
		maxContextTokens: 128000,
	}
}

func (buffer *Buffer) Reset() *Buffer {
	system := buffer.System()
	buffer.messages = make([]provider.Message, 0)
	return buffer.Poke(system)
}

func (buffer *Buffer) System() provider.Message {
	return buffer.messages[0]
}

func (buffer *Buffer) Peek() []provider.Message {
	buffer.truncate()
	return buffer.messages
}

func (buffer *Buffer) Poke(message provider.Message) *Buffer {
	buffer.messages = append(buffer.messages, message)
	return buffer
}

/*
Truncate the buffer to the maximum context tokens, making sure to always keep the
first two messages, which are the system prompt and the user message.
*/
func (buffer *Buffer) truncate() {
	// Always include first two messages (system prompt and user message)
	if len(buffer.messages) < 2 {
		return
	}

	maxTokens := buffer.maxContextTokens - 1024 // Reserve tokens for response
	totalTokens := 0
	var truncatedMessages []provider.Message

	// Add first two messages
	truncatedMessages = append(truncatedMessages, buffer.messages[0], buffer.messages[1])
	totalTokens += buffer.estimateTokens(buffer.messages[0])
	totalTokens += buffer.estimateTokens(buffer.messages[1])

	// Start from the most recent message for the rest
	for i := len(buffer.messages) - 1; i >= 2; i-- {
		msg := buffer.messages[i]
		messageTokens := buffer.estimateTokens(msg)
		if totalTokens+messageTokens <= maxTokens {
			truncatedMessages = append([]provider.Message{msg}, truncatedMessages[2:]...)
			truncatedMessages = append(buffer.messages[0:2], truncatedMessages...)
			totalTokens += messageTokens
		} else {
			break
		}
	}

	buffer.messages = truncatedMessages
}

func (buffer *Buffer) estimateTokens(msg provider.Message) int {
	encoding, err := tiktoken.EncodingForModel("gpt-4o-mini")
	if err != nil {
		log.Error("Error getting encoding", "error", err)
		return 0
	}

	tokensPerMessage := 4 // As per OpenAI's token estimation guidelines

	numTokens := tokensPerMessage
	numTokens += len(encoding.Encode(msg.Content, nil, nil))
	if msg.Role == "user" || msg.Role == "assistant" || msg.Role == "system" || msg.Role == "tool" {
		numTokens += len(encoding.Encode(msg.Role, nil, nil))
	}

	return numTokens
}
