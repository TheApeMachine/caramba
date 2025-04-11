package twoface

import (
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/zeromq/goczmq"
)

type Router struct {
	sock *goczmq.Sock
}

func NewRouter(addr string) (*Router, error) {
	sock, err := goczmq.NewRouter("tcp://0.0.0.0:5555")

	if err != nil {
		return nil, errnie.New(
			errnie.WithError(err),
			errnie.WithMessage("failed to create router"),
		)
	}

	return &Router{
		sock: sock,
	}, nil
}

func (router *Router) Read(p []byte) (n int, err error) {
	return router.sock.Read(p)
}

func (router *Router) Write(p []byte) (n int, err error) {
	return router.sock.Write(p)
}

func (router *Router) Close() error {
	router.sock.Destroy()
	return nil
}

// Send sends a message through the router socket.
func (router *Router) Send(msg []byte) error {
	_, err := router.sock.Write(msg)
	return err
}

// SendMessage sends a multi-part message through the router socket.
func (router *Router) SendMessage(msg [][]byte) error {

	return router.sock.SendMessage(msg)
}
