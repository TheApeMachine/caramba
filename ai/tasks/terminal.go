package tasks

import (
	"io"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
)

type AccumulatorBridge struct {
	accumulator *stream.Accumulator
	buffer      []byte
}

func NewAccumulatorBridge(acc *stream.Accumulator) *AccumulatorBridge {
	return &AccumulatorBridge{
		accumulator: acc,
		buffer:      make([]byte, 0),
	}
}

func (b *AccumulatorBridge) Read(p []byte) (n int, err error) {
	if len(b.buffer) == 0 {
		return 0, io.EOF
	}
	n = copy(p, b.buffer)
	b.buffer = b.buffer[n:]
	return n, nil
}

func (b *AccumulatorBridge) Write(p []byte) (n int, err error) {
	b.accumulator.Write(p)
	return len(p), nil
}

func (b *AccumulatorBridge) Close() error {
	return nil
}

type Terminal struct {
	container *tools.Container
}

func NewTerminal() *Terminal {
	return &Terminal{
		container: tools.NewContainer(),
	}
}

func (task *Terminal) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]any) {
	if err := task.container.Initialize(); err != nil {
		accumulator.Write([]byte(err.Error()))
		return
	}

	bridge := NewAccumulatorBridge(accumulator)
	if err := task.container.Connect(ctx.Identity.Ctx, bridge); err != nil {
		accumulator.Write([]byte(err.Error()))
		return
	}
}
