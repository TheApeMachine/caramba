package twoface

import (
	"io"
	"sync"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/datura"
)

var onceHub sync.Once
var hub *Hub

/*
Hub is a message queue for two-way communication between components.
*/
type Hub struct {
	queue chan *datura.Artifact
	mu    sync.RWMutex
	pipes map[string]*pipe
}

type pipe struct {
	reader *io.PipeReader
	writer *io.PipeWriter
}

/*
NewHub creates a new Hub.
*/
func NewHub() *Hub {
	onceHub.Do(func() {
		hub = &Hub{
			queue: make(chan *datura.Artifact),
			pipes: make(map[string]*pipe),
		}
	})

	return hub
}

/*
NewTransport creates a new bidirectional transport for RPC communication.
Returns an io.ReadWriteCloser that can be used with Cap'n Proto RPC.
*/
func (hub *Hub) NewTransport() io.ReadWriteCloser {
	r, w := io.Pipe()

	hub.mu.Lock()
	id := uuid.New().String()
	hub.pipes[id] = &pipe{
		reader: r,
		writer: w,
	}
	hub.mu.Unlock()

	return &Transport{
		hub: hub,
		id:  id,
		r:   r,
		w:   w,
	}
}

// Transport implements io.ReadWriteCloser for RPC communication
type Transport struct {
	hub *Hub
	id  string
	r   *io.PipeReader
	w   *io.PipeWriter
}

func (transport *Transport) Read(p []byte) (n int, err error) {
	return transport.r.Read(p)
}

func (transport *Transport) Write(p []byte) (n int, err error) {
	return transport.w.Write(p)
}

func (transport *Transport) Close() error {
	transport.hub.mu.Lock()
	delete(transport.hub.pipes, transport.id)
	transport.hub.mu.Unlock()

	transport.r.Close()
	return transport.w.Close()
}

/*
Send sends a message to the Hub.
*/
func (hub *Hub) Send(artifact *datura.Artifact) {
	hub.queue <- artifact
}

/*
Receive receives a message from the Hub.
*/
func (hub *Hub) Receive() *datura.Artifact {
	return <-hub.queue
}
