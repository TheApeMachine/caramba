package core

import (
	"encoding/json"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewMessage tests the NewMessage constructor
func TestNewMessage(t *testing.T) {
	Convey("Given parameters for a new Message", t, func() {
		role := "user"
		name := "testuser"
		content := "test message content"

		Convey("When creating a new Message", func() {
			msg := NewMessage(role, name, content)

			Convey("Then the message should have the correct properties", func() {
				So(msg, ShouldNotBeNil)
				So(msg.Role, ShouldEqual, role)
				So(msg.Name, ShouldEqual, name)
				So(msg.Content, ShouldEqual, content)

				// Verify that buffers are initialized
				So(msg.in, ShouldNotBeNil)
				So(msg.out, ShouldNotBeNil)
				So(msg.enc, ShouldNotBeNil)
				So(msg.dec, ShouldNotBeNil)

				// Verify pre-encoding happened
				So(msg.out.Len(), ShouldBeGreaterThan, 0)
			})
		})
	})
}

// TestMessageRead tests the Read method
func TestMessageRead(t *testing.T) {
	Convey("Given a Message with content", t, func() {
		msg := NewMessage("user", "testuser", "test content")
		buffer := make([]byte, 1024)

		Convey("When reading from the message", func() {
			n, err := msg.Read(buffer)

			Convey("Then it should return valid JSON data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				var parsed MessageData
				err := json.Unmarshal(buffer[:n], &parsed)
				So(err, ShouldBeNil)
				So(parsed.Role, ShouldEqual, "user")
				So(parsed.Name, ShouldEqual, "testuser")
				So(parsed.Content, ShouldEqual, "test content")
			})
		})

		Convey("When reading until empty", func() {
			// First read to consume buffer
			firstBuffer := make([]byte, 1024)
			_, _ = msg.Read(firstBuffer)

			// Second read should be empty
			n, err := msg.Read(buffer)

			Convey("Then it should return EOF", func() {
				So(err, ShouldEqual, io.EOF)
				So(n, ShouldEqual, 0)
			})
		})
	})
}

// TestMessageWrite tests the Write method
func TestMessageWrite(t *testing.T) {
	Convey("Given a Message", t, func() {
		msg := NewMessage("user", "testuser", "original content")

		Convey("When writing valid JSON data", func() {
			newData := &MessageData{
				Role:    "assistant",
				Name:    "ai",
				Content: "new content",
			}
			jsonData, _ := json.Marshal(newData)

			n, err := msg.Write(jsonData)

			Convey("Then it should update the message data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))
				So(msg.Role, ShouldEqual, "assistant")
				So(msg.Name, ShouldEqual, "ai")
				So(msg.Content, ShouldEqual, "new content")
			})

			Convey("And reading should return the updated data", func() {
				buffer := make([]byte, 1024)
				n, err := msg.Read(buffer)

				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				var parsed MessageData
				err = json.Unmarshal(buffer[:n], &parsed)
				So(err, ShouldBeNil)
				So(parsed.Role, ShouldEqual, "assistant")
				So(parsed.Name, ShouldEqual, "ai")
				So(parsed.Content, ShouldEqual, "new content")
			})
		})

		Convey("When writing invalid JSON", func() {
			originalRole := msg.Role
			originalName := msg.Name
			originalContent := msg.Content

			invalidJSON := []byte(`{"role": "invalid" - broken json`)
			n, err := msg.Write(invalidJSON)

			Convey("Then it should retain bytes but not update fields", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
				So(msg.Role, ShouldEqual, originalRole)
				So(msg.Name, ShouldEqual, originalName)
				So(msg.Content, ShouldEqual, originalContent)
			})
		})
	})
}

// TestMessageClose tests the Close method
func TestMessageClose(t *testing.T) {
	Convey("Given a Message with content", t, func() {
		msg := NewMessage("user", "testuser", "test content")

		Convey("When closing the message", func() {
			err := msg.Close()

			Convey("Then it should reset all properties", func() {
				So(err, ShouldBeNil)
				So(msg.Role, ShouldEqual, "")
				So(msg.Name, ShouldEqual, "")
				So(msg.Content, ShouldEqual, "")
			})

			Convey("And the buffers should be empty", func() {
				So(msg.in.Len(), ShouldEqual, 0)
				So(msg.out.Len(), ShouldEqual, 0)

				// Reading should return EOF
				buffer := make([]byte, 1024)
				n, err := msg.Read(buffer)
				So(err, ShouldEqual, io.EOF)
				So(n, ShouldEqual, 0)
			})
		})
	})
}
