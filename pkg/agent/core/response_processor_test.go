package core

import (
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewResponseProcessor(t *testing.T) {
	Convey("Given a need for a response processor", t, func() {
		Convey("When creating a new response processor", func() {
			processor := NewResponseProcessor()

			Convey("Then it should not be nil", func() {
				So(processor, ShouldNotBeNil)
			})
		})
	})
}

func TestProcessChunkContent(t *testing.T) {
	Convey("Given a response processor", t, func() {
		processor := NewResponseProcessor()

		Convey("When processing JSON content with a content field", func() {
			input := `{"content": "Hello world"}`
			result := processor.ProcessChunkContent(input)

			Convey("Then it should extract the content field", func() {
				So(result, ShouldEqual, "Hello world")
			})
		})

		Convey("When processing plain text content", func() {
			input := "Hello world"
			result := processor.ProcessChunkContent(input)

			Convey("Then it should return the original content", func() {
				So(result, ShouldEqual, input)
			})
		})
	})
}

func TestProcessorFormatStreamedContent(t *testing.T) {
	Convey("Given a response processor", t, func() {
		processor := NewResponseProcessor()

		Convey("When formatting streamed content", func() {
			input := "# Header"
			result := processor.FormatStreamedContent(input)

			Convey("Then it should preserve the content", func() {
				// In test environments, color formatting may be disabled
				// Just check that the content is preserved
				So(result, ShouldContainSubstring, input)
			})
		})
	})
}

func TestExtractToolCalls(t *testing.T) {
	Convey("Given a response processor", t, func() {
		processor := NewResponseProcessor()

		Convey("When extracting a single tool call", func() {
			toolCall := ToolCall{
				Name: "testTool",
				Args: map[string]any{"arg1": "value1"},
			}
			toolCallJSON, _ := json.Marshal(toolCall)

			result := processor.ExtractToolCalls(string(toolCallJSON))

			Convey("Then it should extract the tool call correctly", func() {
				So(len(result), ShouldEqual, 1)
				So(result[0].Name, ShouldEqual, "testTool")
				So(result[0].Args["arg1"], ShouldEqual, "value1")
			})
		})

		Convey("When extracting multiple tool calls", func() {
			toolCalls := []ToolCall{
				{Name: "tool1", Args: map[string]any{"arg1": "value1"}},
				{Name: "tool2", Args: map[string]any{"arg2": "value2"}},
			}
			toolCallsJSON, _ := json.Marshal(toolCalls)

			result := processor.ExtractToolCalls(string(toolCallsJSON))

			Convey("Then it should extract all tool calls", func() {
				So(len(result), ShouldEqual, 2)
				So(result[0].Name, ShouldEqual, "tool1")
				So(result[1].Name, ShouldEqual, "tool2")
			})
		})

		Convey("When extracting from invalid JSON", func() {
			result := processor.ExtractToolCalls("not json")

			Convey("Then it should return an empty slice", func() {
				So(len(result), ShouldEqual, 0)
			})
		})
	})
}

func TestSummarizeToolCallArgs(t *testing.T) {
	Convey("Given a response processor", t, func() {
		processor := NewResponseProcessor()

		Convey("When summarizing tool call args", func() {
			toolCall := ToolCall{
				Name: "testTool",
				Args: map[string]any{"arg1": "value1"},
			}

			result := processor.SummarizeToolCallArgs(toolCall)

			Convey("Then it should return a string summary", func() {
				So(result, ShouldNotBeEmpty)
			})
		})
	})
}

func TestFormatToolResult(t *testing.T) {
	Convey("Given a response processor", t, func() {
		processor := NewResponseProcessor()

		Convey("When formatting a string tool result", func() {
			result := processor.FormatToolResult("testTool", "test result")

			Convey("Then it should return the string", func() {
				So(result, ShouldEqual, "test result")
			})
		})

		Convey("When formatting a non-string tool result", func() {
			result := processor.FormatToolResult("testTool", map[string]string{"key": "value"})

			Convey("Then it should return a JSON string", func() {
				So(result, ShouldContainSubstring, "key")
				So(result, ShouldContainSubstring, "value")
			})
		})
	})
}

func TestEstimateTokens(t *testing.T) {
	Convey("Given a response processor", t, func() {
		processor := NewResponseProcessor()

		Convey("When estimating tokens for a string", func() {
			input := "This is a test string with multiple words"
			result := processor.EstimateTokens(input)

			Convey("Then it should return a reasonable estimate", func() {
				So(result, ShouldBeGreaterThan, 0)
				// Roughly 7 words * 4/3 = ~9-10 tokens
				So(result, ShouldBeBetween, 8, 11)
			})
		})

		Convey("When estimating tokens for an empty string", func() {
			result := processor.EstimateTokens("")

			Convey("Then it should return 0", func() {
				So(result, ShouldEqual, 0)
			})
		})
	})
}
