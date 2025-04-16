package stores

import (
	"bytes"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/stores/types"
	"github.com/theapemachine/caramba/pkg/task"
)

func TestSessionRead(t *testing.T) {
	Convey("Given a Store with data", t, func() {
		store := NewMockStore[*task.Task]()
		t := task.NewTask()
		q := types.NewQuery(
			types.WithFilter("id", t.ID),
			types.WithPayload(t),
		)

		store.Poke(q)

		Convey("And a Query", func() {
			query := types.NewQuery(
				types.WithFilter("id", t.ID),
			)

			Convey("When a new session is created", func() {
				session := NewSession(
					WithStore[*task.Task](store),
					WithQuery[*task.Task](query),
					WithModel(task.NewTask()),
				)

				Convey("It should be properly initialized", func() {
					So(session.IsReady(), ShouldBeTrue)
					So(session.instance, ShouldEqual, store)
					So(session.query, ShouldEqual, query)
					So(session.inBuffer, ShouldNotBeNil)
					So(session.outBuffer, ShouldNotBeNil)
				})

				Convey("It should be able to read data", func() {
					buf := bytes.NewBuffer([]byte{})

					n, err := io.Copy(buf, session)
					So(err, ShouldBeNil)
					So(n, ShouldBeGreaterThan, 0)

					tt := task.NewTask()
					nnn, err := io.Copy(tt, buf)
					So(err, ShouldBeNil)
					So(nnn, ShouldBeGreaterThan, 0)
					So(tt.ID, ShouldEqual, t.ID)
				})
			})
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given a Store", t, func() {
		store := NewMockStore[*task.Task]()

		Convey("And a Task to be stored", func() {
			t := task.NewTask()

			Convey("And a Query", func() {
				query := types.NewQuery(
					types.WithFilter("id", t.ID),
					types.WithPayload(t),
				)

				Convey("When a new session is created", func() {
					session := NewSession(
						WithStore[*task.Task](store),
						WithQuery[*task.Task](query),
						WithModel(task.NewTask()),
					)

					Convey("It should be properly initialized", func() {
						So(session.IsReady(), ShouldBeTrue)
						So(session.instance, ShouldEqual, store)
						So(session.query, ShouldEqual, query)
						So(session.inBuffer, ShouldNotBeNil)
						So(session.outBuffer, ShouldNotBeNil)
					})

					Convey("It should be able to write data", func() {
						// Write the task to the session
						n, err := io.Copy(session, t)
						So(err, ShouldBeNil)
						So(n, ShouldBeGreaterThan, 0)

						// Verify the task was stored correctly
						storedTask, err := store.Peek(query)
						So(err, ShouldBeNil)
						So(storedTask, ShouldNotBeNil)

						retrievedTask := task.NewTask()
						nn, err := io.Copy(retrievedTask, storedTask)
						So(err, ShouldBeNil)
						So(nn, ShouldBeGreaterThan, 0)
						So(retrievedTask.ID, ShouldEqual, t.ID)
					})
				})
			})
		})
	})
}
