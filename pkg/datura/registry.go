package datura

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"sync"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var (
	once     sync.Once
	registry *Registry
)

func init() {
	registry = NewRegistry()
}

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
	mu      sync.RWMutex
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

	registry.mu.Lock()
	defer registry.mu.Unlock()

	if builder.ID() == "" {
		errnie.Fatal(errnie.New(
			errnie.WithError(errors.New("id is empty")),
			errnie.WithMessage("no id provided for registration"),
		))
	}

	registry.buffers[builder.ID()] = NewBuffer(builder)
	return builder
}

func (registry *Registry) Unregister(registerable Registerable) {
	errnie.Trace("datura.Unregister", "id", registerable.ID())

	registry.mu.Lock()
	defer registry.mu.Unlock()

	if registerable.ID() == "" {
		errnie.Fatal(errnie.New(
			errnie.WithError(errors.New("id is empty")),
			errnie.WithMessage("no id provided for unregistration"),
		))
	}

	delete(registry.buffers, registerable.ID())
}

func (registry *Registry) Get(registerable Registerable) *Buffer {
	errnie.Trace("datura.Get", "id", registerable.ID())

	registry.mu.RLock()
	buffer, ok := registry.buffers[registerable.ID()]
	registry.mu.RUnlock()

	if ok {
		return buffer
	}

	registry.mu.Lock()
	defer registry.mu.Unlock()

	buffer, ok = registry.buffers[registerable.ID()]

	if ok {
		return buffer
	}

	if registerable.ID() == "" {
		errnie.Fatal(errnie.New(
			errnie.WithError(errors.New("id is empty during registration attempt")),
			errnie.WithMessage("no id provided for registration attempt"),
		))
	}

	newBuffer := NewBuffer(registerable)
	registry.buffers[registerable.ID()] = newBuffer
	return newBuffer
}
