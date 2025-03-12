package core

import (
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
				So(msg.Buffer, ShouldNotBeNil)
			})
		})
	})
}
