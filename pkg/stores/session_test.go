package stores

import (
	"bytes"
	"errors"
	"io"
	"sync"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stores/types"
	"github.com/theapemachine/caramba/pkg/task"
)

type MockStore struct {
	*errnie.Error
	*errnie.State
	data *sync.Map
}

func NewMockstore() *MockStore {
	return &MockStore{
		Error: errnie.NewError(),
		State: errnie.NewState().To(errnie.StateReady),
		data:  new(sync.Map),
	}
}

func (m *MockStore) Peek(query *types.Query) (io.Reader, error) {
	if value, ok := m.data.Load(query.Filters["id"]); ok {
		if task, ok := value.(*task.Task); ok {
			return task, nil
		}
		return nil, errors.New("value is not a *task.Task")
	}

	return nil, errors.New("not ok")
}

func (m *MockStore) Poke(query *types.Query) error {
	m.data.Store(query.Filters["id"], query.Payload)
	return nil
}

func TestRead(t *testing.T) {
	Convey("Given a Store with data", t, func() {
		store := NewMockstore()
		t := task.NewTask()
		store.data.Store(t.ID, t)

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
		store := NewMockstore()

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
