package core

import (
	"encoding/json"
	"errors"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewEvent tests the NewEvent constructor
func TestNewEvent(t *testing.T) {
	Convey("Given parameters for a new Event", t, func() {
		message := NewMessage("user", "testuser", "test content")
		testError := errors.New("test error")

		Convey("When creating a new Event", func() {
			event := NewEvent(message, testError)

			Convey("Then the event should have the correct properties", func() {
				So(event, ShouldNotBeNil)
				So(event.Message, ShouldEqual, message)
				So(event.Error, ShouldEqual, testError)
				So(event.ToolCalls, ShouldNotBeNil)
				So(len(event.ToolCalls), ShouldEqual, 0)

				// Verify that buffers are initialized
				So(event.in, ShouldNotBeNil)
				So(event.out, ShouldNotBeNil)
				So(event.enc, ShouldNotBeNil)
				So(event.dec, ShouldNotBeNil)

				// Verify pre-encoding happened
				So(event.out.Len(), ShouldBeGreaterThan, 0)
			})
		})

		Convey("When creating an Event without an error", func() {
			event := NewEvent(message, nil)

			Convey("Then the event should have nil error", func() {
				So(event.Error, ShouldBeNil)
			})
		})

		Convey("When creating an Event without a message", func() {
			event := NewEvent(nil, testError)

			Convey("Then the event should have nil message", func() {
				So(event.Message, ShouldBeNil)
			})
		})
	})
}

// TestEventRead tests the Read method of Event
func TestEventRead(t *testing.T) {
	Convey("Given an Event with data", t, func() {
		message := NewMessage("user", "testuser", "test content")
		testError := errors.New("test error")
		event := NewEvent(message, testError)
		buffer := make([]byte, 1024)

		Convey("When reading from the event", func() {
			n, err := event.Read(buffer)

			Convey("Then it should return valid JSON data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				// We can't directly unmarshal into EventData because of the error field
				// So we'll check that it contains our message data
				jsonStr := string(buffer[:n])
				So(jsonStr, ShouldContainSubstring, "test content")
				So(jsonStr, ShouldContainSubstring, "user")
				So(jsonStr, ShouldContainSubstring, "testuser")
			})
		})

		Convey("When reading until empty", func() {
			// First read to consume buffer
			firstBuffer := make([]byte, 1024)
			_, _ = event.Read(firstBuffer)

			// Second read should be empty
			n, err := event.Read(buffer)

			Convey("Then it should return EOF", func() {
				So(err, ShouldEqual, io.EOF)
				So(n, ShouldEqual, 0)
			})
		})
	})
}

// TestEventWrite tests the Write method of Event
func TestEventWrite(t *testing.T) {
	Convey("Given an Event", t, func() {
		originalMessage := NewMessage("user", "testuser", "original content")
		event := NewEvent(originalMessage, nil)

		Convey("When writing new data", func() {
			newMessage := NewMessage("assistant", "ai", "response content")
			newEvent := &EventData{
				Message:   newMessage,
				ToolCalls: []*ToolCall{},
				Error:     nil,
			}

			jsonData, err := json.Marshal(newEvent)
			So(err, ShouldBeNil)

			n, err := event.Write(jsonData)

			Convey("Then it should update the event data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))
				So(event.Message, ShouldNotEqual, originalMessage)
				So(event.Message.Role, ShouldEqual, "assistant")
				So(event.Message.Name, ShouldEqual, "ai")
				So(event.Message.Content, ShouldEqual, "response content")
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"message": {"role": "invalid" - broken json`)
			n, err := event.Write(invalidJSON)

			Convey("Then it should retain bytes but not update fields", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
				So(event.Message, ShouldEqual, originalMessage)
			})
		})
	})
}

// TestEventClose tests the Close method of Event
func TestEventClose(t *testing.T) {
	Convey("Given an Event with content", t, func() {
		message := NewMessage("user", "testuser", "test content")
		testError := errors.New("test error")
		event := NewEvent(message, testError)

		Convey("When closing the event", func() {
			err := event.Close()

			Convey("Then it should reset all properties", func() {
				So(err, ShouldBeNil)
				So(event.Message, ShouldBeNil)
				So(event.ToolCalls, ShouldBeNil)
				So(event.Error, ShouldBeNil)
			})
		})
	})
}

// TestEventWithToolCalls tests the WithToolCalls method
func TestEventWithToolCalls(t *testing.T) {
	Convey("Given an Event", t, func() {
		event := NewEvent(NewMessage("user", "testuser", "test content"), nil)

		Convey("When adding tool calls", func() {
			toolCall1 := NewToolCall("id1", "tool1", map[string]any{"param1": "value1"})
			toolCall2 := NewToolCall("id2", "tool2", map[string]any{"param2": "value2"})

			event.WithToolCalls(toolCall1, toolCall2)

			Convey("Then the tool calls should be added", func() {
				So(len(event.ToolCalls), ShouldEqual, 2)
				So(event.ToolCalls[0], ShouldEqual, toolCall1)
				So(event.ToolCalls[1], ShouldEqual, toolCall2)
			})

			Convey("When adding more tool calls", func() {
				toolCall3 := NewToolCall("id3", "tool3", map[string]any{"param3": "value3"})
				event.WithToolCalls(toolCall3)

				Convey("Then the new tool call should be appended", func() {
					So(len(event.ToolCalls), ShouldEqual, 3)
					So(event.ToolCalls[2], ShouldEqual, toolCall3)
				})
			})
		})
	})
}
