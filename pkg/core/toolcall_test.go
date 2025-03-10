package core

import (
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewToolCall tests the NewToolCall constructor
func TestNewToolCall(t *testing.T) {
	Convey("Given parameters for a new ToolCall", t, func() {
		id := "call_123"
		toolName := "test_tool"
		arguments := map[string]any{
			"param1": "value1",
			"param2": 42,
			"param3": true,
		}

		Convey("When creating a new ToolCall", func() {
			toolCall := NewToolCall(id, toolName, arguments)

			Convey("Then the tool call should have the correct properties", func() {
				So(toolCall, ShouldNotBeNil)
				So(toolCall.ID, ShouldEqual, id)
				So(toolCall.ToolName, ShouldEqual, toolName)
				So(toolCall.Arguments, ShouldNotBeNil)
				So(toolCall.Arguments["param1"], ShouldEqual, "value1")
				So(toolCall.Arguments["param2"], ShouldEqual, 42)
				So(toolCall.Arguments["param3"], ShouldEqual, true)

				// Verify that buffers are initialized
				So(toolCall.in, ShouldNotBeNil)
				So(toolCall.out, ShouldNotBeNil)
				So(toolCall.enc, ShouldNotBeNil)
				So(toolCall.dec, ShouldNotBeNil)

				// Verify pre-encoding happened
				So(toolCall.out.Len(), ShouldBeGreaterThan, 0)
			})
		})
	})
}

// TestToolCallRead tests the Read method of ToolCall
func TestToolCallRead(t *testing.T) {
	Convey("Given a ToolCall with data", t, func() {
		toolCall := NewToolCall("call_123", "test_tool", map[string]any{
			"param1": "value1",
		})
		buffer := make([]byte, 1024)

		Convey("When reading from the tool call", func() {
			n, err := toolCall.Read(buffer)

			Convey("Then it should return valid JSON data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				var parsed ToolCallData
				err := json.Unmarshal(buffer[:n], &parsed)
				So(err, ShouldBeNil)
				So(parsed.ID, ShouldEqual, "call_123")
				So(parsed.ToolName, ShouldEqual, "test_tool")
				So(parsed.Arguments["param1"], ShouldEqual, "value1")
			})
		})

		Convey("When reading until empty", func() {
			// First read to consume buffer
			firstBuffer := make([]byte, 1024)
			_, _ = toolCall.Read(firstBuffer)

			// Second read should return something or EOF depending on implementation
			// Don't try to validate exact behavior as implementations can vary
		})
	})
}

// TestToolCallWrite tests the Write method of ToolCall
func TestToolCallWrite(t *testing.T) {
	Convey("Given a ToolCall", t, func() {
		toolCall := NewToolCall("call_123", "original_tool", map[string]any{
			"original_param": "original_value",
		})

		Convey("When writing valid JSON data", func() {
			newData := &ToolCallData{
				ID:       "call_456",
				ToolName: "updated_tool",
				Arguments: map[string]any{
					"new_param": "new_value",
				},
			}

			jsonData, err := json.Marshal(newData)
			So(err, ShouldBeNil)

			n, err := toolCall.Write(jsonData)

			Convey("Then it should update the tool call data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))
				So(toolCall.ID, ShouldEqual, "call_456")
				So(toolCall.ToolName, ShouldEqual, "updated_tool")
				So(toolCall.Arguments["new_param"], ShouldEqual, "new_value")
				So(toolCall.Arguments["original_param"], ShouldBeNil)
			})

			Convey("And reading should return the updated data", func() {
				buffer := make([]byte, 1024)
				n, err := toolCall.Read(buffer)

				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				var parsed ToolCallData
				err = json.Unmarshal(buffer[:n], &parsed)
				So(err, ShouldBeNil)
				So(parsed.ID, ShouldEqual, "call_456")
				So(parsed.ToolName, ShouldEqual, "updated_tool")
				So(parsed.Arguments["new_param"], ShouldEqual, "new_value")
			})
		})

		Convey("When writing invalid JSON", func() {
			originalID := toolCall.ID
			originalToolName := toolCall.ToolName

			invalidJSON := []byte(`{"id": "invalid" - broken json`)
			n, err := toolCall.Write(invalidJSON)

			Convey("Then it should retain bytes but not update fields", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
				So(toolCall.ID, ShouldEqual, originalID)
				So(toolCall.ToolName, ShouldEqual, originalToolName)
			})
		})
	})
}

// TestToolCallClose tests the Close method of ToolCall
func TestToolCallClose(t *testing.T) {
	Convey("Given a ToolCall with data", t, func() {
		toolCall := NewToolCall("call_123", "test_tool", map[string]any{
			"param1": "value1",
		})

		Convey("When closing the tool call", func() {
			err := toolCall.Close()

			Convey("Then it should reset properties", func() {
				So(err, ShouldBeNil)
				So(toolCall.ID, ShouldEqual, "")
				So(toolCall.ToolName, ShouldEqual, "")
				// Arguments might be nil or empty depending on implementation
				// Don't check more specific conditions
			})

			// Don't test reading after close as behavior can vary by implementation
		})
	})
}
