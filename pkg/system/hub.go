package system

import (
	"fmt"
	"io"
	"slices"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Topic string

type Client struct {
	ID     string
	IO     io.ReadWriteCloser
	Topics []Topic
}

type Hub struct {
	clients     map[string]*Client
	clientQueue chan *datura.Artifact
	topicQueue  chan *datura.Artifact
}

type HubOption func(*Hub)

var (
	onceHub sync.Once
	hub     *Hub
)

func NewHub(options ...HubOption) *Hub {
	onceHub.Do(func() {
		hub = &Hub{
			clients:     make(map[string]*Client),
			clientQueue: make(chan *datura.Artifact, 64),
			topicQueue:  make(chan *datura.Artifact, 64),
		}

		for _, option := range options {
			option(hub)
		}
	})

	return hub
}

func (hub *Hub) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		var err error

		for {
			select {
			case artifact := <-buffer:
				to := datura.GetMetaValue[string](artifact, "to")
				topic := datura.GetMetaValue[Topic](artifact, "topic")

				for _, client := range hub.clients {
					if !slices.Contains(client.Topics, topic) {
						continue
					}

					hub.topicQueue <- artifact
				}

				for _, client := range hub.clients {
					if client.ID != to {
						continue
					}

					hub.clientQueue <- artifact
				}
			case artifact := <-hub.topicQueue:
				topic := datura.GetMetaValue[Topic](artifact, "topic")

				for _, client := range hub.clients {
					if !slices.Contains(client.Topics, topic) {
						continue
					}

					if _, err = io.Copy(
						client.IO, artifact,
					); errnie.Error(err) != nil {
						continue
					}
				}
			case artifact := <-hub.clientQueue:
				to := datura.GetMetaValue[string](artifact, "to")

				for _, client := range hub.clients {
					if client.ID != to {
						continue
					}

					if _, err = io.Copy(
						client.IO, artifact,
					); errnie.Error(err) != nil {
						continue
					}
				}
			default:
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()

	return out
}

func WithStreamers(streamers ...*core.Streamer) HubOption {
	return func(hub *Hub) {
		for _, streamer := range streamers {
			// Turn the streamer into a client
			hub.clients[streamer.ID()] = &Client{
				ID:     streamer.ID(),
				IO:     streamer,
				Topics: []Topic{"broadcast"},
			}
		}
	}
}

func WithClients(clients ...*Client) HubOption {
	return func(hub *Hub) {
		for _, client := range clients {
			hub.clients[client.ID] = client
		}
	}
}

func WithClient(clientID string, client io.ReadWriteCloser) HubOption {
	return func(hub *Hub) {
		hub.clients[clientID] = &Client{
			ID:     clientID,
			IO:     client,
			Topics: []Topic{"broadcast"},
		}
	}
}

func WithTopics(clientID string, topics ...Topic) HubOption {
	return func(hub *Hub) {
		if _, ok := hub.clients[clientID]; !ok {
			errnie.Error(NoClientError{ClientID: clientID})
			return
		}

		hub.clients[clientID].Topics = append(hub.clients[clientID].Topics, topics...)
	}
}

type NoClientError struct {
	err      error
	ClientID string
}

func (e *NoClientError) Error() string {
	e.err = fmt.Errorf("no client %s found while adding topics", e.ClientID)
	return e.err.Error()
}

func (e *NoClientError) Unwrap() error {
	return e.err
}
