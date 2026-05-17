package devteam

import (
	"testing"

	openai "github.com/openai/openai-go"
	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/config"
)

func TestNewOpenAIProvider(test *testing.T) {
	Convey("Given provider configuration for OpenAI-compatible endpoints", test, func() {
		Convey("It should use chat completions when a custom base URL is configured", func() {
			provider := NewOpenAIProvider(config.ProviderConfig{
				Provider: "ollama",
				BaseURL:  "http://127.0.0.1:11434/v1",
				Model:    "llama",
			})

			So(provider.useChatCompletions, ShouldBeTrue)
		})

		Convey("It should use responses for the default OpenAI endpoint", func() {
			provider := NewOpenAIProvider(config.ProviderConfig{
				Provider: "openai",
				Model:    "gpt-4o",
			})

			So(provider.useChatCompletions, ShouldBeFalse)
		})
	})
}

func TestOpenAIProvider_buildChatMessages(test *testing.T) {
	Convey("Given provider-neutral chat history with tool calls", test, func() {
		provider := &OpenAIProvider{}
		request := ChatRequest{
			System: "system prompt",
			Messages: []ChatMessage{
				{Role: "user", Content: "work"},
				{
					Role: "assistant",
					ToolCalls: []ToolCall{{
						ID:    "call-1",
						Name:  "done",
						Input: map[string]any{"test_file": "agent_test.go"},
					}},
				},
				{Role: "tool", ToolCallID: "call-1", Content: "ok"},
			},
		}

		Convey("It should map roles and tool calls into chat completion messages", func() {
			messages, err := provider.buildChatMessages(request)

			So(err, ShouldBeNil)
			So(messages, ShouldHaveLength, 4)
			So(messages[0].OfSystem, ShouldNotBeNil)
			So(messages[2].OfAssistant.ToolCalls, ShouldHaveLength, 1)
			So(messages[3].OfTool.ToolCallID, ShouldEqual, "call-1")
		})
	})
}

func TestOpenAIProvider_parseChatCompletion(test *testing.T) {
	Convey("Given a chat completion with tool calls and usage", test, func() {
		provider := &OpenAIProvider{}
		completion := &openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{
					Content: "ready",
					ToolCalls: []openai.ChatCompletionMessageToolCall{{
						ID: "call-1",
						Function: openai.ChatCompletionMessageToolCallFunction{
							Name:      "done",
							Arguments: `{"test_file":"agent_test.go"}`,
						},
					}},
				},
			}},
			Usage: openai.CompletionUsage{
				PromptTokens:     3,
				CompletionTokens: 5,
				TotalTokens:      8,
			},
		}

		Convey("It should return provider-neutral content, tool calls, and token counts", func() {
			response, err := provider.parseChatCompletion(completion)

			So(err, ShouldBeNil)
			So(response.Content, ShouldEqual, "ready")
			So(response.ToolCalls, ShouldHaveLength, 1)
			So(response.ToolCalls[0].Name, ShouldEqual, "done")
			So(response.InputTokens, ShouldEqual, 3)
			So(response.OutputTokens, ShouldEqual, 5)
			So(response.TotalTokens, ShouldEqual, 8)
		})
	})
}

func BenchmarkOpenAIProvider_buildChatMessages(benchmark *testing.B) {
	provider := &OpenAIProvider{}
	request := ChatRequest{
		System:   "system prompt",
		Messages: []ChatMessage{{Role: "user", Content: "work"}},
	}

	for benchmark.Loop() {
		_, _ = provider.buildChatMessages(request)
	}
}
