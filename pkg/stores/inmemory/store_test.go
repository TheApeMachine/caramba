package inmemory

import (
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/stores/types"
	"github.com/theapemachine/caramba/pkg/task"
)

func TestPeek(t *testing.T) {
	Convey("Given a new store", t, func() {
		store := NewStore[*task.Task]()

		Convey("When it has data", func() {
			task1 := task.NewTask()
			q1 := types.NewQuery(
				types.WithFilter("id", task1.ID),
				types.WithPayload(task1),
			)
			store.Poke(q1)

			Convey("It should be able to Peek", func() {
				testTask1 := task.NewTask()
				testQ1 := types.NewQuery(
					types.WithFilter("id", q1.Filters["id"]),
				)

				r, err := store.Peek(testQ1)
				So(err, ShouldBeNil)
				So(r, ShouldNotBeNil)

				n, err := io.Copy(testTask1, r)
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)
				So(testTask1.ID, ShouldEqual, task1.ID)
			})
		})
	})
}
