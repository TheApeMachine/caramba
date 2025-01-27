package tasks

import (
	"io"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/tools"
)

// IOBridge is a wrapper around any io.ReadWriteCloser that lets the Agent read/write to it.
type IOBridge struct {
	container *tools.Container
	Conn      io.ReadWriteCloser
}

func (b *IOBridge) Start() {
	// Start the container if it's not already running
	if err := b.container.Start(); err != nil {
		log.Error("Error starting container", "error", err)
	}

}

func (b *IOBridge) Read(p []byte) (n int, err error) {
	return b.Conn.Read(p)
}

func (b *IOBridge) Write(p []byte) (n int, err error) {
	return b.Conn.Write(p)
}

func (b *IOBridge) Close() error {
	return b.Conn.Close()
}
