package ai

import (
	"encoding/json"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/core"
)

// TestNewContext tests the NewContext constructor
func TestNewContext(t *testing.T) {
	Convey("Given no parameters", t, func() {
		Convey("When creating a new Context", func() {
			ctx := NewContext()

			Convey("Then the context should have the correct properties", func() {
				So(ctx, ShouldNotBeNil)
				So(ctx.ContextData, ShouldNotBeNil)
				So(ctx.Messages, ShouldNotBeNil)
				So(ctx.Tools, ShouldNotBeNil)
				So(ctx.buffer, ShouldNotBeNil)
				So(ctx.dec, ShouldNotBeNil)
				So(ctx.enc, ShouldNotBeNil)
				So(ctx.Stream, ShouldBeTrue)
			})
		})
	})
}

// TestContextRead tests the Read method of Context
func TestContextRead(t *testing.T) {
	Convey("Given a Context with encoded data", t, func() {
		ctx := NewContext()
		buffer := make([]byte, 1024)

		Convey("When reading from the context", func() {
			n, err := ctx.Read(buffer)

			Convey("Then it should return valid JSON data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				// Verify it contains ContextData structure
				jsonStr := string(buffer[:n])
				So(jsonStr, ShouldContainSubstring, "messages")
				So(jsonStr, ShouldContainSubstring, "model")
				So(jsonStr, ShouldContainSubstring, "tools")
			})
		})

		Convey("When reading from an empty buffer", func() {
			// Create a new context but with an empty output buffer
			emptyCtx := NewContext()
			emptyCtx.buffer = nil

			// Try to read from the empty buffer
			emptyBuffer := make([]byte, 1024)
			n, err := emptyCtx.Read(emptyBuffer)

			Convey("Then it should return EOF", func() {
				So(err, ShouldEqual, io.EOF)
				So(n, ShouldEqual, 0)
			})
		})
	})
}

// TestContextWrite tests the Write method of Context
func TestContextWrite(t *testing.T) {
	Convey("Given a Context", t, func() {
		ctx := NewContext()

		Convey("When writing valid context data", func() {
			// Create a new context data with specific values
			message := core.NewMessage("user", "testuser", "test content")
			newCtxData := &ContextData{
				Model:    "gpt-4",
				Messages: []*core.Message{message},
				Tools:    make([]*core.Tool, 0),
			}

			// Serialize the context data to JSON
			jsonData, err := json.Marshal(newCtxData)
			So(err, ShouldBeNil)

			n, err := ctx.Write(jsonData)

			Convey("Then it should update the context data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))
				So(ctx.Model, ShouldEqual, "gpt-4")
				So(len(ctx.Messages), ShouldEqual, 1)
				So(ctx.Messages[0].Role, ShouldEqual, "user")
				So(ctx.Messages[0].Content, ShouldEqual, "test content")
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"model": "broken-json`)
			n, err := ctx.Write(invalidJSON)

			Convey("Then it should not return an error but not update the context", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
				So(ctx.Model, ShouldNotEqual, "broken-json")
			})
		})

		Convey("When the output buffer is reset during write", func() {
			// First, write valid data to update the context
			validData := &ContextData{
				Model: "test-model",
			}
			jsonData, _ := json.Marshal(validData)

			// Create a fresh context
			freshCtx := NewContext()
			// Make sure the buffer has content
			So(freshCtx.buffer, ShouldNotBeNil)

			// Now write to it, which should reset the buffer and then re-encode
			freshCtx.Write(jsonData)

			// Check the output buffer has content after re-encoding
			So(freshCtx.buffer, ShouldNotBeNil)
		})
	})
}

// TestContextClose tests the Close method of Context
func TestContextClose(t *testing.T) {
	Convey("Given a Context with data", t, func() {
		ctx := NewContext()

		// Add some data to the context
		ctx.Model = "test-model"
		ctx.Messages = append(ctx.Messages, core.NewMessage("user", "testuser", "test content"))
		ctx.Tools = append(ctx.Tools, core.NewTool("test-tool", "description", nil))

		Convey("When closing the context", func() {
			err := ctx.Close()

			Convey("Then it should clear the buffers but not reset the content fields", func() {
				So(err, ShouldBeNil)
				// The Close method only resets buffers, not field contents
				So(ctx.buffer, ShouldBeNil)
				So(ctx.dec, ShouldBeNil)
				So(ctx.enc, ShouldBeNil)
				// These won't be changed by Close
				So(ctx.Model, ShouldEqual, "test-model")
				So(len(ctx.Messages), ShouldEqual, 1)
				So(len(ctx.Tools), ShouldEqual, 1)
			})
		})
	})
}
