package subscription

import (
	"io"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/stores/types"
	"github.com/theapemachine/caramba/pkg/task"
)

func TestRead(t *testing.T) {
	t.Parallel()

	Convey("Given a Subscription", t, func() {
		t := task.NewTask()
		sub := NewSubscription(WithID(t.ID))
		newTask := &task.Task{}

		store := NewMockStore(
			WithSubscription[*task.Task](sub),
		)

		go func() {
			isPolling := true

			for isPolling {
				// Wait for the subscription to be ready
				if _, err := io.Copy(newTask, sub); err != nil {
					return
				}

				if newTask.ID != t.ID {
					isPolling = false
				}

				time.Sleep(100 * time.Millisecond)
			}
		}()

		Convey("When reading from the subscription", func() {
			// Use the store to trigger a notification
			query := types.NewQuery(
				types.WithFilter("id", t.ID),
				types.WithPayload(t),
			)
			err := store.Poke(query)
			So(err, ShouldBeNil)
			time.Sleep(100 * time.Millisecond)
			So(newTask.ID, ShouldEqual, t.ID)
		})
	})
}
