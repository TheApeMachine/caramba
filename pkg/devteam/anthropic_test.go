package devteam

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/config"
)

func TestAnthropicProviderChat(test *testing.T) {
	Convey("Given an Anthropic provider with an empty system prompt", test, func() {
		var requestBody string
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			raw, err := io.ReadAll(request.Body)
			if err != nil {
				test.Errorf("read request body: %v", err)
				return
			}

			requestBody = string(raw)

			writer.Header().Set("Content-Type", "application/json")
			_, err = writer.Write([]byte(`{
				"id":"msg_test",
				"type":"message",
				"role":"assistant",
				"model":"claude-test",
				"content":[{"type":"text","text":"ok"}],
				"stop_reason":"end_turn",
				"stop_sequence":null,
				"usage":{"input_tokens":2,"output_tokens":3}
			}`))
			if err != nil {
				test.Errorf("write response: %v", err)
			}
		}))
		defer server.Close()

		provider := NewAnthropicProvider(config.ProviderConfig{
			Provider: "anthropic",
			APIKey:   "test-key",
			BaseURL:  server.URL,
			Model:    "claude-test",
		})

		Convey("It should omit the system field and preserve usage", func() {
			response, err := provider.Chat(context.Background(), ChatRequest{
				Messages: []ChatMessage{{Role: "user", Content: "hello"}},
			})

			So(err, ShouldBeNil)
			So(strings.Contains(requestBody, `"system"`), ShouldBeFalse)
			So(response.Content, ShouldEqual, "ok")
			So(response.InputTokens, ShouldEqual, 2)
			So(response.OutputTokens, ShouldEqual, 3)
			So(response.TotalTokens, ShouldEqual, 5)
		})
	})
}
