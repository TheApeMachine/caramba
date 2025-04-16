package stores

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stores/types"
)

// Session is a generic interface to a store that implements io.ReadWriteCloser
// It can be used with any type that can be encoded/decoded via JSON
type Session[T any] struct {
	*errnie.Error
	*errnie.State
	instance  types.Store
	query     *types.Query
	encoder   *json.Encoder
	decoder   *json.Decoder
	inBuffer  *bytes.Buffer
	outBuffer *bytes.Buffer
	model     any
}

type SessionOptions[T any] func(*Session[T])

// NewSession creates a new session with the given options
func NewSession[T any](opts ...SessionOptions[T]) *Session[T] {
	inBuffer := bytes.NewBuffer([]byte{})
	outBuffer := bytes.NewBuffer([]byte{})

	session := &Session[T]{
		Error:     errnie.NewError(),
		State:     errnie.NewState().To(errnie.StateReady),
		inBuffer:  inBuffer,
		outBuffer: outBuffer,
		encoder:   json.NewEncoder(outBuffer),
		decoder:   json.NewDecoder(inBuffer),
	}

	for _, opt := range opts {
		opt(session)
	}

	return session
}

// Read implements io.ReadCloser.
func (session *Session[T]) Read(p []byte) (n int, err error) {
	if !session.OK() {
		return 0, session.Error
	}

	if session.query == nil {
		return 0, session.Add(errnie.Validation(nil, "missing query"))
	}

	if session.IsReady() {
		session.State.To(errnie.StateBusy)
		item, err := session.instance.Peek(session.query)

		if err != nil {
			session.State.To(errnie.StateFailed)
			return 0, session.Add(errnie.IO(err, "failed to read item"))
		}

		session.outBuffer.Reset()

		if err := session.encoder.Encode(item); err != nil {
			return 0, session.Add(errnie.IO(err, "failed to encode item"))
		}
	}

	if n, err = session.outBuffer.Read(p); err != nil && err != io.EOF {
		session.State.To(errnie.StateFailed)
		return 0, session.Add(errnie.IO(err, "failed to read from buffer"))
	}

	if n == 0 || err == io.EOF {
		session.State.To(errnie.StateReady)
		return 0, io.EOF
	}

	return
}

// Write implements io.WriteCloser.
func (session *Session[T]) Write(p []byte) (n int, err error) {
	if session.IsReady() {
		session.State.To(errnie.StateBusy)
		session.inBuffer.Reset()
	}

	if n, err = session.inBuffer.Write(p); err != nil && err != io.EOF {
		return 0, session.Add(errnie.IO(err, "failed to write to buffer"))
	}

	session.decoder.Decode(session.model)

	// Call the store's Poke method to store the data
	if err = session.instance.Poke(session.query); err != nil {
		session.State.To(errnie.StateFailed)
		return n, session.Add(errnie.IO(err, "failed to store data"))
	}

	session.State.To(errnie.StateReady)

	return n, nil
}

// Close implements io.Closer.
func (session *Session[T]) Close() error {
	session.State.To(errnie.StateDone)
	session.inBuffer.Reset()
	session.outBuffer.Reset()

	return nil
}

func WithStore[T any](store types.Store) SessionOptions[T] {
	return func(session *Session[T]) {
		session.instance = store
	}
}

func WithQuery[T any](query *types.Query) SessionOptions[T] {
	return func(session *Session[T]) {
		session.query = query
	}
}

func WithModel[T any](model T) SessionOptions[T] {
	return func(session *Session[T]) {
		session.model = model
	}
}
