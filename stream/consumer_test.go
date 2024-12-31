package stream

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/provider"
)

func TestConsumer(t *testing.T) {
	convey.Convey("Given a new Consumer", t, func() {
		consumer := NewConsumer()

		convey.Convey("When processing a simple key-value pair", func() {
			stream := make(chan provider.Event)
			done := make(chan bool)

			go func() {
				consumer.Print(stream, true)
				done <- true
			}()

			stream <- provider.Event{Text: `{"name": "test"}`}
			close(stream)
			<-done

			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
			convey.So(consumer.indent, convey.ShouldEqual, 0)
		})

		convey.Convey("When processing nested objects", func() {
			stream := make(chan provider.Event)
			done := make(chan bool)

			go func() {
				consumer.Print(stream, true)
				done <- true
			}()

			stream <- provider.Event{Text: `{
				"user": {
					"name": "test",
					"age": 25
				}
			}`}
			close(stream)
			<-done

			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
			convey.So(consumer.indent, convey.ShouldEqual, 0)
		})

		convey.Convey("When processing arrays", func() {
			stream := make(chan provider.Event)
			done := make(chan bool)

			go func() {
				consumer.Print(stream, true)
				done <- true
			}()

			stream <- provider.Event{Text: `{
				"items": ["one", "two", "three"]
			}`}
			close(stream)
			<-done

			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
			convey.So(consumer.indent, convey.ShouldEqual, 0)
		})

		convey.Convey("When processing escaped characters", func() {
			stream := make(chan provider.Event)
			done := make(chan bool)

			go func() {
				consumer.Print(stream, true)
				done <- true
			}()

			stream <- provider.Event{Text: `{"message": "Hello \"World\""}`}
			close(stream)
			<-done

			convey.So(consumer.state, convey.ShouldEqual, StateUndetermined)
			convey.So(consumer.indent, convey.ShouldEqual, 0)
		})
	})
}
