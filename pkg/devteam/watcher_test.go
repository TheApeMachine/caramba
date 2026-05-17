package devteam

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestColumnEventJSON(t *testing.T) {
	Convey("Given a raw kanban_column_notify JSON payload", t, func() {
		payload := `{
			"id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
			"research_project_id": "a1b2c3d4-0000-0000-0000-000000000001",
			"column_key": "todo",
			"old_column_key": "backlog",
			"title": "Add dark mode",
			"description": "Support dark theme across all pages."
		}`

		Convey("It should decode into a ColumnEvent", func() {
			var event ColumnEvent
			err := json.Unmarshal([]byte(payload), &event)

			So(err, ShouldBeNil)
			So(event.ID, ShouldEqual, "f47ac10b-58cc-4372-a567-0e02b2c3d479")
			So(event.ColumnKey, ShouldEqual, "todo")
			So(event.OldColumnKey, ShouldEqual, "backlog")
			So(event.Title, ShouldEqual, "Add dark mode")
		})
	})
}

func TestWatcherConstruction(t *testing.T) {
	Convey("Given a valid database URL", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		watcher := NewWatcher(ctx, "postgres://localhost/test")

		Convey("It should expose a readable events channel", func() {
			So(watcher.Events(), ShouldNotBeNil)
		})
	})
}

func TestWatcher_publishError(t *testing.T) {
	Convey("Given a watcher error channel", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		watcher := NewWatcher(ctx, "postgres://localhost/test")

		Convey("It should publish reconnect errors for the orchestrator to observe", func() {
			watcher.publishError(errors.New("listener reconnect"))

			err := <-watcher.Errors()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "listener reconnect")
		})
	})
}

func BenchmarkColumnEventJSON(b *testing.B) {
	payload := []byte(`{
		"id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
		"research_project_id": "a1b2c3d4-0000-0000-0000-000000000001",
		"column_key": "todo",
		"old_column_key": "backlog",
		"title": "Add dark mode",
		"description": "Support dark theme across all pages."
	}`)

	for b.Loop() {
		var event ColumnEvent
		_ = json.Unmarshal(payload, &event)
	}
}
