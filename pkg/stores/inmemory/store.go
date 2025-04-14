package inmemory

import (
	"bytes"
	"encoding/json"
	"io"
	"sync"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stores/types"
	"github.com/theapemachine/caramba/pkg/task"
)

var (
	once     sync.Once
	instance *Store
)

/*
Store is a simple in-memory implementation of task.TaskStore,
following our common store interface.
*/
type Store struct {
	*errnie.Error
	data *sync.Map
}

func NewStore() *Store {
	errnie.Trace("inmemmory.NewStore")

	once.Do(func() {
		instance = &Store{
			Error: errnie.NewError(),
			data:  new(sync.Map),
		}
	})

	return instance
}

type Session struct {
	*errnie.Error
	*errnie.State
	instance  *Store
	query     *types.Query
	encoder   *json.Encoder
	decoder   *json.Decoder
	inBuffer  *bytes.Buffer
	outBuffer *bytes.Buffer
}

func NewSession(query *types.Query) *Session {
	if instance == nil {
		NewStore()
	}

	outBuffer := bytes.NewBuffer([]byte{})
	inBuffer := bytes.NewBuffer([]byte{})

	return &Session{
		Error:     errnie.NewError(),
		State:     errnie.NewState().To(errnie.StateReady),
		instance:  instance,
		query:     query,
		encoder:   json.NewEncoder(outBuffer),
		decoder:   json.NewDecoder(inBuffer),
		inBuffer:  inBuffer,
		outBuffer: outBuffer,
	}
}

// Read implements io.ReadCloser.
func (session *Session) Read(p []byte) (n int, err error) {
	if !session.instance.OK() {
		return 0, session.instance.Error
	}

	if session.query == nil {
		return 0, session.Add(errnie.Validation(nil, "missing query"))
	}

	if session.IsReady() {
		session.State.To(errnie.StateBusy)
		taskItem, ok := session.instance.data.Load(session.query.Filters["id"])

		if !ok {
			session.State.To(errnie.StateFailed)
			return 0, session.Add(errnie.IO(err, "failed to read task"))
		}

		task := taskItem.(*task.Task)
		session.outBuffer.Reset()
		session.encoder.Encode(task)
	}

	if n, err = session.outBuffer.Read(p); err != nil && err != io.EOF {
		session.State.To(errnie.StateFailed)
		return 0, session.Add(errnie.IO(err, "failed to read task"))
	}

	if n == 0 || err == io.EOF {
		session.State.To(errnie.StateReady)
		return 0, io.EOF
	}

	return
}

// Write implements io.WriteCloser.
func (session *Session) Write(p []byte) (n int, err error) {
	if !session.instance.OK() {
		return 0, session.instance.Error
	}

	if session.IsReady() {
		session.State.To(errnie.StateBusy)
		session.inBuffer.Reset()
	}

	if n, err = session.inBuffer.Write(p); err != nil && err != io.EOF {
		return 0, session.Add(errnie.IO(err, "failed to write task"))
	}

	session.State.To(errnie.StateReady)
	task := &task.Task{}

	if err = session.decoder.Decode(task); err != nil {
		return 0, session.Add(errnie.IO(err, "failed to decode task"))
	}

	session.instance.data.Store(task.ID, task)

	return

}

// Close implements io.Closer.
func (session *Session) Close() error {
	session.State.To(errnie.StateDone)
	session.inBuffer.Reset()
	session.outBuffer.Reset()
	session.encoder = nil
	session.decoder = nil
	session.instance = nil
	session.query = nil
	return nil
}
