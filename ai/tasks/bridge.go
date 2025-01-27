package tasks

import (
	"io"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/tools"
)

// IOBridge is a wrapper around any io.ReadWriteCloser that lets the Agent read/write to it.
type IOBridge struct {
	container *tools.Container
	Conn      io.ReadWriteCloser
}

func (b *IOBridge) Start() {

}

func (b *IOBridge) Execute(cmd string) string {
	// Handle exit command specially
	if cmd == "exit" {
		b.Close()
		return "Terminal session ended.\n"
	}

	log.Info("Executing command", "command", cmd)

	// Execute command and ensure output ends with newline
	output := b.container.ExecuteCommand(cmd)
	if !strings.HasSuffix(output, "\n") {
		output += "\n"
	}

	return output
}

func (b *IOBridge) Read(p []byte) (n int, err error) {
	return b.Conn.Read(p)
}

func (b *IOBridge) Write(p []byte) (n int, err error) {
	return b.Conn.Write(p)
}

func (b *IOBridge) Close() error {
	if b.Conn != nil {
		err := b.Conn.Close()
		b.Conn = nil
		return err
	}
	return nil
}
