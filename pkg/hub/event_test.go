package hub

import (
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewEvent(t *testing.T) {
	Convey("Given parameters for a new event", t, func() {
		origin := "test-origin"
		topic := "test-topic"
		role := "test-role"
		eventType := EventTypeInfo
		message := "test message"
		meta := map[string]string{"key": "value"}

		Convey("When NewEvent is called", func() {
			event := NewEvent(origin, topic, role, eventType, message, meta)

			Convey("Then an event with the correct properties should be created", func() {
				So(event, ShouldNotBeNil)
				So(event.Origin, ShouldEqual, origin)
				So(event.Topic, ShouldEqual, topic)
				So(event.Role, ShouldEqual, role)
				So(event.Type, ShouldEqual, eventType)
				So(event.Message, ShouldEqual, message)
				So(event.Meta, ShouldResemble, meta)
			})
		})
	})
}

func TestEventString(t *testing.T) {
	Convey("Given an event", t, func() {
		event := NewEvent("origin", "topic", "role", EventTypeInfo, "test message", nil)

		Convey("When String is called", func() {
			result := event.String()

			Convey("Then it should return a string with type and message", func() {
				So(result, ShouldEqual, "info: test message")
			})
		})
	})
}

func TestEventJSON(t *testing.T) {
	Convey("Given an event with all fields populated", t, func() {
		event := NewEvent("origin", "topic", "role", EventTypeInfo, "test message", map[string]string{"key": "value"})

		Convey("When JSON is called", func() {
			jsonStr := event.JSON()

			Convey("Then it should return a valid JSON string", func() {
				So(jsonStr, ShouldNotBeEmpty)

				// Verify JSON can be parsed
				var parsedEvent map[string]interface{}
				err := json.Unmarshal([]byte(jsonStr), &parsedEvent)
				So(err, ShouldBeNil)

				// Check fields
				So(parsedEvent["Origin"], ShouldEqual, "origin")
				So(parsedEvent["Topic"], ShouldEqual, "topic")
				So(parsedEvent["Role"], ShouldEqual, "role")
				So(parsedEvent["Type"], ShouldEqual, "info")
				So(parsedEvent["Message"], ShouldEqual, "test message")
				meta, ok := parsedEvent["Meta"].(map[string]interface{})
				So(ok, ShouldBeTrue)
				So(meta["key"], ShouldEqual, "value")
			})
		})
	})
}

func TestSpecializedEventConstructors(t *testing.T) {
	Convey("Given parameters for specialized event constructors", t, func() {
		origin := "test-origin"
		topic := "test-topic"
		role := "test-role"
		message := "test message"

		Convey("When NewError is called", func() {
			event := NewError(origin, topic, role, message)

			Convey("Then an Error event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeError)
				So(event.Origin, ShouldEqual, origin)
				So(event.Topic, ShouldEqual, topic)
				So(event.Role, ShouldEqual, role)
				So(event.Message, ShouldEqual, message)
			})
		})

		Convey("When NewWarning is called", func() {
			event := NewWarning(origin, topic, role, message)

			Convey("Then a Warning event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeWarning)
			})
		})

		Convey("When NewInfo is called", func() {
			event := NewInfo(origin, topic, role, message)

			Convey("Then an Info event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeInfo)
			})
		})

		Convey("When NewDebug is called", func() {
			event := NewDebug(origin, topic, role, message)

			Convey("Then a Debug event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeDebug)
			})
		})

		Convey("When NewFatal is called", func() {
			event := NewFatal(origin, topic, role, message)

			Convey("Then a Fatal event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeFatal)
			})
		})

		Convey("When NewLog is called", func() {
			event := NewLog(origin, topic, role, message)

			Convey("Then a Log event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeLog)
			})
		})

		Convey("When NewStatus is called", func() {
			event := NewStatus(origin, role, message)

			Convey("Then a Status event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeStatus)
				So(event.Topic, ShouldEqual, "ui")
			})
		})

		Convey("When NewMessage is called", func() {
			event := NewMessage(origin, message)

			Convey("Then a Message event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeMessage)
				So(event.Topic, ShouldEqual, "ui")
				So(event.Role, ShouldEqual, "agent")
			})
		})

		Convey("When NewChunk is called", func() {
			event := NewChunk(origin, message)

			Convey("Then a Chunk event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeChunk)
				So(event.Topic, ShouldEqual, "ui")
				So(event.Role, ShouldEqual, "agent")
			})
		})

		Convey("When NewToolCall is called", func() {
			event := NewToolCall(origin, role, message)

			Convey("Then a ToolCall event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeToolCall)
				So(event.Topic, ShouldEqual, "metrics")
			})
		})

		Convey("When NewMetric is called", func() {
			event := NewMetric(origin, topic, role, message)

			Convey("Then a Metric event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeMetric)
			})
		})

		Convey("When NewClear is called", func() {
			event := NewClear(origin, topic, role, message)

			Convey("Then a Clear event should be created", func() {
				So(event.Type, ShouldEqual, EventTypeClear)
			})
		})
	})
}
