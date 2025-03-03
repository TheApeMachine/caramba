package hub

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewEvent(t *testing.T) {
	Convey("Given parameters for a new event", t, func() {
		origin := "test-origin"
		topic := "test-topic"
		eventType := EventTypeInfo
		message := "test message"
		meta := map[string]string{"key": "value"}

		Convey("When NewEvent is called", func() {
			event := &Event{
				Origin:  origin,
				Topic:   TopicTypeLog,
				Type:    eventType,
				Message: message,
				Meta:    meta,
			}

			Convey("Then an event with the correct properties should be created", func() {
				So(event, ShouldNotBeNil)
				So(event.Origin, ShouldEqual, origin)
				So(event.Topic, ShouldEqual, topic)
				So(event.Type, ShouldEqual, eventType)
				So(event.Message, ShouldEqual, message)
				So(event.Meta, ShouldResemble, meta)
			})
		})
	})
}
