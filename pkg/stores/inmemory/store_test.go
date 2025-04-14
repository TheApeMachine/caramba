package inmemory

import (
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/stores/types"
	"github.com/theapemachine/caramba/pkg/task"
)

func TestStore(t *testing.T) {
	Convey("Given a new store", t, func() {
		store := NewStore()

		Convey("It should be empty", func() {
			count := 0

			store.data.Range(func(key, value any) bool {
				count++
				return true
			})

			So(count, ShouldEqual, 0)
		})
	})
}

func TestNewSession(t *testing.T) {
	Convey("Given a new store", t, func() {
		NewStore()

		Convey("And a query with an id filter", func() {
			query := types.NewQuery(
				types.WithFilter("id", "1234567890"),
			)

			Convey("When a new session is created", func() {
				session := NewSession(query)

				Convey("It should be ready", func() {
					So(session.IsReady(), ShouldBeTrue)
				})
			})
		})
	})
}

func TestRead(t *testing.T) {
	Convey("Given a new store with a task", t, func() {
		store := NewStore()
		testTask := task.NewTask()
		store.data.Store(testTask.ID, testTask)

		Convey("And a query with an id filter", func() {
			query := types.NewQuery(
				types.WithFilter("id", testTask.ID),
			)

			Convey("When a new session is created", func() {
				session := NewSession(query)

				Convey("And the session is read", func() {
					newTask := task.NewTask()
					n, err := io.Copy(newTask, session)

					So(err, ShouldBeNil)
					So(n, ShouldNotEqual, 0)
				})
			})
		})
	})
}
