package tasks

import (
	"io"
)

// IOBridge is a wrapper around any io.ReadWriteCloser that lets the Agent read/write to it.
type IOBridge struct {
	Conn io.ReadWriteCloser
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
