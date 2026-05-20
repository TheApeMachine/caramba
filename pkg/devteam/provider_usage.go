package devteam

import (
	"time"

	"github.com/theapemachine/qpool"
)

func publishChatUsage(role string, started time.Time, response ChatResponse) {
	qpool.Publish(qpool.NewInfoEvent(
		"devteam",
		"llm.chat",
		"provider chat completed",
		[]qpool.Field{
			{Key: "role", Value: role},
			{Key: "duration_ms", Value: time.Since(started).Milliseconds()},
			{Key: "input_tokens", Value: response.InputTokens},
			{Key: "output_tokens", Value: response.OutputTokens},
			{Key: "total_tokens", Value: response.TotalTokens},
		},
	))
}
