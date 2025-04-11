package twoface

import (
	"errors"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/zeromq/goczmq"
)

type Transport struct {
	sock    *goczmq.Sock
	hubAddr string
}

// NewTransport creates a new Dealer socket connected to the specified Hub address.
func NewTransport(name string) (*Transport, error) {
	hubAddr := viper.GetViper().GetString("settings.hub.address")
	errnie.Info("NewTransport", "hub_addr", hubAddr)

	sock, err := goczmq.NewDealer(hubAddr)

	if err != nil {
		return nil, errnie.InternalError(err)
	}

	// Generate a unique identity for this socket
	identity := name
	sock.SetIdentity(identity)

	errnie.Info("transport created", "identity", identity, "hub_addr", hubAddr)

	return &Transport{
		sock:    sock,
		hubAddr: hubAddr,
	}, nil
}

func (transport *Transport) Sock() *goczmq.Sock {
	return transport.sock
}

func (transport *Transport) Subscribe(topics []string) (err error) {
	errnie.Trace("Transport.Subscribe", "topics", topics)

	identity := transport.sock.Identity()

	if identity == "" {
		return errnie.InternalError(errors.New("transport has no identity"))
	}

	errnie.Info(
		"subscribe",
		"topics", topics,
		"identity", identity,
	)

	artifact := datura.New(
		datura.WithRole(datura.ArtifactRoleSubscriber),
		datura.WithIssuer(identity),
		datura.WithMetadata(map[string]any{
			"type":   "subscribe",
			"topics": topics,
		}),
	)

	msg := [][]byte{
		artifact.Bytes(),
	}

	if err := transport.sock.SendMessage(msg); err != nil {
		return errnie.InternalError(err)
	}

	response, err := transport.sock.RecvMessage()

	if err != nil {
		return errnie.InternalError(err)
	}

	errnie.Info("response", "frames", len(response))

	for i, frame := range response {
		frameContent := "<empty>"

		if len(frame) > 0 {
			frameContent = string(frame)
		}

		errnie.Info("  [Frame]", "index", i, "content", frameContent)
	}

	responseArtifact := datura.New(
		datura.WithBytes(response[1]),
	)

	if responseArtifact.HasError() {
		return errnie.InternalError(responseArtifact)
	}

	if responseArtifact.ActsAs(datura.ArtifactRoleAcknowledger) {
		errnie.Info("subscription ACK received")
		return nil
	}

	return nil
}

func (transport *Transport) Publish(artifact *datura.Artifact) error {
	errnie.Trace("Transport.Publish", "artifact", artifact)

	msg := [][]byte{
		artifact.Bytes(),
	}

	if err := transport.sock.SendMessage(msg); err != nil {
		return errnie.InternalError(err)
	}

	return nil
}

func (transport *Transport) Recv() (msg [][]byte, err error) {
	errnie.Trace("Transport.Recv")

	if msg, err = transport.sock.RecvMessage(); err != nil {
		return nil, errnie.InternalError(err)
	}

	return msg, nil
}

func (transport *Transport) Send(msg [][]byte) error {
	errnie.Trace("Transport.Send", "msg", msg)

	if err := transport.sock.SendMessage(msg); err != nil {
		return errnie.InternalError(err)
	}

	return nil
}

func (transport *Transport) Close() error {
	errnie.Trace("Transport.Close")

	if transport.sock != nil {
		transport.sock.Destroy()
	}

	return nil
}
