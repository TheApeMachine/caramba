package datura

import (
	"bufio"
	"bytes"
	"io"
	"sync"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var (
	once     sync.Once
	registry *Registry
)

type Registerable interface {
	io.ReadWriteCloser
	ID() string
	Message() *capnp.Message
}

type Buffer struct {
	registerable Registerable
	Encoder      *capnp.Encoder
	Decoder      *capnp.Decoder
	Buffer       *bufio.ReadWriter
	State        errnie.State
}

func NewBuffer(registerable Registerable) *Buffer {
	errnie.Trace("datura.NewBuffer")

	shared := bytes.NewBuffer(nil)
	buffer := bufio.NewReadWriter(
		bufio.NewReader(shared),
		bufio.NewWriter(shared),
	)

	return &Buffer{
		registerable: registerable,
		Encoder:      capnp.NewEncoder(buffer),
		Decoder:      capnp.NewDecoder(buffer),
		Buffer:       buffer,
		State:        errnie.StateUnknown,
	}
}

type Registry struct {
	buffers map[string]*Buffer
}

func NewRegistry() *Registry {
	errnie.Trace("datura.NewRegistry")

	once.Do(func() {
		registry = &Registry{
			buffers: make(map[string]*Buffer),
		}
	})

	return registry
}

func Register[T Registerable](builder T) T {
	errnie.Trace("datura.Register", "id", builder.ID())

	NewRegistry().buffers[builder.ID()] = NewBuffer(builder)
	return builder
}

func (registry *Registry) Unregister(id string) {
	errnie.Trace("datura.Unregister")

	delete(registry.buffers, id)
}

func (registry *Registry) Get(id string) *Buffer {
	errnie.Trace("datura.Get", "id", id)

	return registry.buffers[id]
}
