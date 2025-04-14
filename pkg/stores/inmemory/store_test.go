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
			So(store.Error, ShouldNotBeNil)
			So(store.OK(), ShouldBeTrue)
		})

		Convey("When multiple tasks are stored", func() {
			task1 := task.NewTask()
			task2 := task.NewTask()
			task3 := task.NewTask()

			store.data.Store(task1.ID, task1)
			store.data.Store(task2.ID, task2)
			store.data.Store(task3.ID, task3)

			Convey("It should contain all of them", func() {
				count := 0

				store.data.Range(func(key, value any) bool {
					count++
					return true
				})

				So(count, ShouldEqual, 3)

				for _, t := range []*task.Task{task1, task2, task3} {
					storedTask, exists := store.data.Load(t.ID)
					So(exists, ShouldBeTrue)
					So(storedTask, ShouldEqual, t)
				}
			})
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
					So(session.Error, ShouldNotBeNil)
					So(session.OK(), ShouldBeTrue)
					So(session.encoder, ShouldNotBeNil)
					So(session.decoder, ShouldNotBeNil)
					So(session.instance, ShouldEqual, instance)
					So(session.query, ShouldEqual, query)
					So(session.inBuffer, ShouldNotBeNil)
					So(session.outBuffer, ShouldNotBeNil)
				})
			})
		})

		Convey("And a query without filters", func() {
			query := types.NewQuery()

			Convey("When a new session is created", func() {
				session := NewSession(query)

				Convey("It should be properly initialized", func() {
					So(session.IsReady(), ShouldBeTrue)
					So(session.query, ShouldEqual, query)
					So(session.query.Filters, ShouldNotBeNil)
					So(len(session.query.Filters), ShouldEqual, 0)
				})
			})
		})
	})
}

func TestRead(t *testing.T) {
	Convey("Given a new store with a task", t, func() {
		store := NewStore()
		testTask := task.NewTask()
		testTask.Metadata["name"] = "Test Read Task"
		testTask.Metadata["description"] = "Task for testing read functionality"
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
					So(newTask.ID, ShouldEqual, testTask.ID)
					So(newTask.Metadata["name"], ShouldEqual, testTask.Metadata["name"])
					So(newTask.Metadata["description"], ShouldEqual, testTask.Metadata["description"])
				})
			})
		})

		Convey("And a query with a non-existent id", func() {
			query := types.NewQuery(
				types.WithFilter("id", "non-existent-id"),
			)

			Convey("When a new session is created", func() {
				session := NewSession(query)

				Convey("And the session is read", func() {
					newTask := task.NewTask()
					n, err := io.Copy(newTask, session)

					So(err, ShouldNotBeNil)
					So(n, ShouldEqual, 0)
					So(session.State.IsFailed(), ShouldBeTrue)
				})
			})
		})

		Convey("And a query without an id filter", func() {
			query := types.NewQuery()

			Convey("When a new session is created", func() {
				session := NewSession(query)

				Convey("And the session is read", func() {
					newTask := task.NewTask()
					n, err := io.Copy(newTask, session)

					So(err, ShouldNotBeNil)
					So(n, ShouldEqual, 0)
				})
			})
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given a new store", t, func() {
		store := NewStore()

		Convey("And a new task", func() {
			testTask := task.NewTask()
			testTask.Metadata["name"] = "Test Write Task"

			Convey("When a new session is created", func() {
				query := types.NewQuery(
					types.WithFilter("id", testTask.ID),
				)
				session := NewSession(query)

				Convey("And the task is written to the session", func() {
					n, err := io.Copy(session, testTask)

					So(err, ShouldBeNil)
					So(n, ShouldNotEqual, 0)

					Convey("Then the task should be stored in the data map", func() {
						var storedTask any
						var exists bool

						storedTask, exists = store.data.Load(testTask.ID)

						So(exists, ShouldBeTrue)
						So(storedTask, ShouldNotBeNil)

						actualTask, ok := storedTask.(*task.Task)
						So(ok, ShouldBeTrue)
						So(actualTask.ID, ShouldEqual, testTask.ID)
						So(actualTask.Metadata["name"], ShouldEqual, testTask.Metadata["name"])
					})
				})
			})
		})
	})
}

func TestClose(t *testing.T) {
	Convey("Given a new store", t, func() {
		store := NewStore()

		Convey("When a new session is created", func() {
			session := NewSession(types.NewQuery())

			Convey("And the session is closed", func() {
				err := session.Close()

				So(err, ShouldBeNil)
				So(session.State.IsDone(), ShouldBeTrue)
				So(session.inBuffer.Len(), ShouldEqual, 0)
				So(session.outBuffer.Len(), ShouldEqual, 0)
				So(session.encoder, ShouldBeNil)
				So(session.decoder, ShouldBeNil)
				So(session.instance, ShouldBeNil)
				So(session.query, ShouldBeNil)
			})
		})

		Convey("When a session is created and used for reading", func() {
			testTask := task.NewTask()
			store.data.Store(testTask.ID, testTask)

			query := types.NewQuery(
				types.WithFilter("id", testTask.ID),
			)
			session := NewSession(query)

			newTask := task.NewTask()
			_, _ = io.Copy(newTask, session)

			Convey("And then closed", func() {
				err := session.Close()

				So(err, ShouldBeNil)
				So(session.State.IsDone(), ShouldBeTrue)
			})
		})

		Convey("When a session is created and used for writing", func() {
			testTask := task.NewTask()
			session := NewSession(types.NewQuery())

			_, _ = io.Copy(session, testTask)

			Convey("And then closed", func() {
				err := session.Close()

				So(err, ShouldBeNil)
				So(session.State.IsDone(), ShouldBeTrue)
			})
		})
	})
}
