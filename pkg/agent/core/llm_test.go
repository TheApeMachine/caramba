package core

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// Define custom error types for testing
type ErrLLMGeneration struct {
	Message string
}

func (e ErrLLMGeneration) Error() string {
	return fmt.Sprintf("LLM generation error: %s", e.Message)
}

type ErrToolExecution struct {
	ToolName string
	Message  string
}

func (e ErrToolExecution) Error() string {
	return fmt.Sprintf("Tool execution error for %s: %s", e.ToolName, e.Message)
}

// MockOpenAILLM is a mock implementation of OpenAI LLM provider for testing
type MockOpenAILLM struct {
	MockLLM
}

func NewOpenAILLM() *MockOpenAILLM {
	return &MockOpenAILLM{}
}

func (m *MockOpenAILLM) Name() string {
	return "OpenAI"
}

func TestLLMMessage(t *testing.T) {
	Convey("Given parameters for a new LLM message", t, func() {
		Convey("When creating a new user message", func() {
			message := LLMMessage{
				Role:    "user",
				Content: "Hello",
			}

			Convey("Then it should have the correct role and content", func() {
				So(message.Role, ShouldEqual, "user")
				So(message.Content, ShouldEqual, "Hello")
			})
		})

		Convey("When creating a new assistant message", func() {
			message := LLMMessage{
				Role:    "assistant",
				Content: "Hello, how can I help?",
			}

			Convey("Then it should have the correct role and content", func() {
				So(message.Role, ShouldEqual, "assistant")
				So(message.Content, ShouldEqual, "Hello, how can I help?")
			})
		})

		Convey("When creating a new system message", func() {
			message := LLMMessage{
				Role:    "system",
				Content: "You are a helpful assistant",
			}

			Convey("Then it should have the correct role and content", func() {
				So(message.Role, ShouldEqual, "system")
				So(message.Content, ShouldEqual, "You are a helpful assistant")
			})
		})
	})
}

func TestLLMParams(t *testing.T) {
	Convey("Given parameters for LLM generation", t, func() {
		Convey("When creating a new set of parameters", func() {
			params := LLMParams{
				Model:       "gpt-4o-mini",
				Temperature: 0.7,
				MaxTokens:   1024,
				Messages: []LLMMessage{
					{Role: "system", Content: "You are a helpful assistant"},
					{Role: "user", Content: "Hello"},
				},
				Tools: []Tool{&MockTool{}},
			}

			Convey("Then it should have the correct values", func() {
				So(params.Model, ShouldEqual, "gpt-4o-mini")
				So(params.Temperature, ShouldEqual, 0.7)
				So(params.MaxTokens, ShouldEqual, 1024)
				So(len(params.Messages), ShouldEqual, 2)
				So(len(params.Tools), ShouldEqual, 1)
			})
		})
	})
}

func TestLLMResponse(t *testing.T) {
	Convey("Given an LLM response", t, func() {
		Convey("When creating a content response", func() {
			response := LLMResponse{
				Type:    ResponseTypeContent,
				Content: "Hello, world!",
			}

			Convey("Then it should have the correct type and content", func() {
				So(response.Type, ShouldEqual, ResponseTypeContent)
				So(response.Content, ShouldEqual, "Hello, world!")
				So(response.Error, ShouldBeNil)
			})
		})

		Convey("When creating a tool call response", func() {
			response := LLMResponse{
				Type: ResponseTypeToolCall,
				ToolCalls: []ToolCall{
					{
						Name: "testTool",
						Args: map[string]any{"arg1": "value1"},
					},
				},
			}

			Convey("Then it should have the correct type and tool calls", func() {
				So(response.Type, ShouldEqual, ResponseTypeToolCall)
				So(len(response.ToolCalls), ShouldEqual, 1)
				So(response.ToolCalls[0].Name, ShouldEqual, "testTool")
				So(response.ToolCalls[0].Args["arg1"], ShouldEqual, "value1")
			})
		})

		Convey("When creating an error response", func() {
			errMsg := "Test error message"
			response := LLMResponse{
				Error: ErrLLMGeneration{Message: errMsg},
			}

			Convey("Then it should have the error set", func() {
				So(response.Error, ShouldNotBeNil)
				So(response.Error.Error(), ShouldContainSubstring, errMsg)
			})
		})
	})
}

func TestToolCall(t *testing.T) {
	Convey("Given a tool call", t, func() {
		Convey("When creating a new tool call", func() {
			toolCall := ToolCall{
				Name: "testTool",
				Args: map[string]any{
					"arg1": "value1",
					"arg2": 42,
					"arg3": true,
				},
			}

			Convey("Then it should have the correct name and arguments", func() {
				So(toolCall.Name, ShouldEqual, "testTool")
				So(len(toolCall.Args), ShouldEqual, 3)
				So(toolCall.Args["arg1"], ShouldEqual, "value1")
				So(toolCall.Args["arg2"], ShouldEqual, 42)
				So(toolCall.Args["arg3"], ShouldEqual, true)
			})
		})
	})
}

func TestErrLLMGeneration(t *testing.T) {
	Convey("Given an LLM generation error", t, func() {
		Convey("When creating a new error", func() {
			errMsg := "Failed to generate response"
			err := ErrLLMGeneration{Message: errMsg}

			Convey("Then it should implement the error interface", func() {
				var e error = err
				So(e, ShouldNotBeNil)
				So(e.Error(), ShouldContainSubstring, errMsg)
			})
		})
	})
}

func TestErrToolExecution(t *testing.T) {
	Convey("Given a tool execution error", t, func() {
		Convey("When creating a new error", func() {
			toolName := "testTool"
			errMsg := "Failed to execute tool"
			err := ErrToolExecution{ToolName: toolName, Message: errMsg}

			Convey("Then it should implement the error interface", func() {
				var e error = err
				So(e, ShouldNotBeNil)
				So(e.Error(), ShouldContainSubstring, toolName)
				So(e.Error(), ShouldContainSubstring, errMsg)
			})
		})
	})
}

// TestOpenAILLM can only test interface compliance since it requires external API calls
func TestOpenAILLM(t *testing.T) {
	Convey("Given a need for an OpenAI LLM provider", t, func() {
		Convey("When creating a new OpenAI LLM", func() {
			llm := NewOpenAILLM()

			Convey("Then it should implement the LLMProvider interface", func() {
				var provider LLMProvider = llm
				So(provider, ShouldNotBeNil)
			})

			Convey("Then it should have the correct name", func() {
				So(llm.Name(), ShouldEqual, "OpenAI")
			})
		})
	})
}
