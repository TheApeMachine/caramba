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
	protocols   map[string]*core.Protocol
}

type HubOption func(*Hub)

var (
	onceHub sync.Once
	hub     *Hub
)

func NewHub(options ...HubOption) *Hub {
	onceHub.Do(func() {
		errnie.Debug("system.NewHub")

		hub = &Hub{
			clients:     make(map[string]*Client),
			clientQueue: make(chan *datura.Artifact, 64),
			topicQueue:  make(chan *datura.Artifact, 64),
			protocols:   make(map[string]*core.Protocol),
		}

		for _, option := range options {
			option(hub)
		}
	})

	return hub
}

func (hub *Hub) RegisterProtocol(protocol *core.Protocol) *core.Protocol {
	errnie.Debug("system.Hub.RegisterProtocol", "protocol", protocol.ID)

	hub.protocols[protocol.ID] = protocol
	return protocol
}

func (hub *Hub) GetProtocol(id string) *core.Protocol {
	errnie.Debug("system.Hub.GetProtocol", "id", id)
	return hub.protocols[id]
}

func (hub *Hub) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("system.Hub.Generate")

	out := make(chan *datura.Artifact, 64)

	for _, client := range hub.clients {
		go func() {
			errnie.Debug("system.Hub.Generate.Read", "client", client.ID)
			clientOut := datura.New()
			var err error

			if _, err = io.Copy(clientOut, client.IO); errnie.Error(err) != nil {
				return
			}

			buffer <- clientOut
		}()
	}

	go func() {
		defer close(out)

		var err error

		for {
			select {
			case artifact := <-buffer:
				to := datura.GetMetaValue[string](artifact, "to")
				topic := Topic(datura.GetMetaValue[string](artifact, "topic"))

				errnie.Debug(
					"Hub received artifact",
					"topic", topic,
					"to", to,
					"role", artifact.Role(),
					"scope", artifact.Scope(),
				)

				// Route by topic first
				topicClients := 0
				for _, client := range hub.clients {
					if !slices.Contains(client.Topics, topic) {
						continue
					}

					errnie.Info("Routing to topic subscriber", "client", client.ID, "topic", topic)
					select {
					case hub.topicQueue <- artifact:
						topicClients++
					default:
						errnie.Error("Topic queue full", "topic", topic)
					}
				}

				// Then route by direct client ID
				if to != "" {
					clientFound := false
					for _, client := range hub.clients {
						if client.ID != to {
							continue
						}

						errnie.Info("Routing to client", "client", client.ID)
						select {
						case hub.clientQueue <- artifact:
							clientFound = true
						default:
							errnie.Error("Client queue full", "client", client.ID)
						}
						break
					}

					if !clientFound {
						errnie.Debug("Client not found", "to", to)
					}
				}

			case artifact := <-hub.topicQueue:
				topic := Topic(datura.GetMetaValue[string](artifact, "topic"))
				errnie.Debug("Processing topic queue", "topic", topic)

				for _, client := range hub.clients {
					if !slices.Contains(client.Topics, topic) {
						continue
					}

					errnie.Debug("Copying to topic subscriber", "client", client.ID, "topic", topic)
					if _, err = io.Copy(
						client.IO, artifact,
					); errnie.Error(err) != nil {
						errnie.Debug("Failed to copy to client", "client", client.ID, "error", err)
						continue
					}
				}

			case artifact := <-hub.clientQueue:
				to := datura.GetMetaValue[string](artifact, "to")
				errnie.Debug("Processing client queue", "to", to)

				for _, client := range hub.clients {
					if client.ID != to {
						continue
					}

					errnie.Debug("Copying to client", "client", client.ID)
					if _, err = io.Copy(
						client.IO, artifact,
					); errnie.Error(err) != nil {
						errnie.Debug("Failed to copy to client", "error", err)
						continue
					}
				}

			case <-time.After(100 * time.Millisecond):
				// Do nothing
			}
		}
	}()

	return out
}

func WithStreamers(streamers ...*core.Streamer) HubOption {
	return func(hub *Hub) {
		for _, streamer := range streamers {
			// Turn the streamer into a client
			client := &Client{
				ID:     streamer.ID(),
				IO:     streamer,
				Topics: []Topic{"broadcast"},
			}

			// Convert string topics to Topic type
			for _, topic := range streamer.Topics() {
				client.Topics = append(client.Topics, Topic(topic))
			}

			errnie.Debug("system.Hub.WithStreamers", "client", client.ID, "topics", client.Topics)
			hub.clients[client.ID] = client
		}
	}
}

func WithClients(clients ...*Client) HubOption {
	return func(hub *Hub) {
		for _, client := range clients {
			client.Topics = append(client.Topics, Topic("broadcast"))
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
