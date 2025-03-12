package core

import (
	"errors"
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
