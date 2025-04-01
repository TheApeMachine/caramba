package core

import (
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/system"
)

type Streamer struct {
	hub    *system.Hub
	buffer *stream.Buffer
}

type StreamerOption func(*Streamer)

func NewStreamer(generator stream.Generator) *Streamer {
	return &Streamer{
		hub: system.NewHub(),
		buffer: stream.NewBuffer(
			stream.WithGenerator(generator),
		),
	}
}

func (streamer *Streamer) Read(p []byte) (n int, err error) {
	return streamer.buffer.Read(p)
}

func (streamer *Streamer) Write(p []byte) (n int, err error) {
	return streamer.buffer.Write(p)
}

func (streamer *Streamer) Close() error {
	return streamer.buffer.Close()
}
