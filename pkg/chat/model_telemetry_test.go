package chat

import (
	"errors"
	"testing"

	"github.com/theapemachine/caramba/pkg/qpool"

	. "github.com/smartystreets/goconvey/convey"
)

func TestModelStartupTelemetry_Publish(test *testing.T) {
	Convey("Given model startup telemetry", test, func() {
		events := captureQPoolEvents()
		defer events.restore()

		telemetry := newModelStartupTelemetry(ModelConfig{
			Manifest:  "model/llm/gpt2.yml",
			Model:     "openai-community/gpt2",
			Tokenizer: "openai-community/gpt2",
			Backend:   "metal",
		})

		Convey("It should publish startup events with model context", func() {
			telemetry.Publish("manifest.resolve", "resolving manifest")

			So(events.values, ShouldHaveLength, 1)
			So(events.values[0].Component, ShouldEqual, "chat")
			So(events.values[0].Op, ShouldEqual, "manifest.resolve")
			So(events.values[0].Message, ShouldEqual, "resolving manifest")
			So(eventField(events.values[0], "manifest"), ShouldEqual, "model/llm/gpt2.yml")
			So(eventField(events.values[0], "model"), ShouldEqual, "openai-community/gpt2")
			So(eventField(events.values[0], "tokenizer"), ShouldEqual, "openai-community/gpt2")
			So(eventField(events.values[0], "backend"), ShouldEqual, "metal")
		})
	})
}

func TestModelStartupTelemetry_Error(test *testing.T) {
	Convey("Given model startup telemetry errors", test, func() {
		events := captureQPoolEvents()
		defer events.restore()

		telemetry := newModelStartupTelemetry(ModelConfig{
			Manifest: "model/llm/gpt2.yml",
		})

		Convey("It should publish errors and return the original value", func() {
			expected := errors.New("boom")
			err := telemetry.Error("weights.resolve", "failed", expected)

			So(err, ShouldEqual, expected)
			So(events.values, ShouldHaveLength, 1)
			So(events.values[0].Err, ShouldEqual, expected)
			So(events.values[0].Op, ShouldEqual, "weights.resolve")
		})
	})
}

type capturedQPoolEvents struct {
	restore func()
	values  []qpool.Event
}

func captureQPoolEvents() *capturedQPoolEvents {
	events := &capturedQPoolEvents{}
	previous := qpool.Publish

	qpool.Publish = func(event qpool.Event) {
		events.values = append(events.values, event)
	}

	events.restore = func() {
		qpool.Publish = previous
	}

	return events
}

func eventField(event qpool.Event, key string) any {
	for _, field := range event.Fields {
		if field.Key == key {
			return field.Value
		}
	}

	return nil
}
