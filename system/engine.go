package system

import (
	"context"
	"io"
)

type Engine struct {
	ctx    context.Context
	cancel context.CancelFunc
	pr     *io.PipeReader
	pw     *io.PipeWriter
}

func NewEngine() *Engine {
	pr, pw := io.Pipe()

	return &Engine{
		pr: pr,
		pw: pw,
	}
}

func (engine *Engine) Read(p []byte) (n int, err error) {
	return 0, nil
}

func (engine *Engine) Write(p []byte) (n int, err error) {
	engine.ctx, engine.cancel = context.WithCancel(context.Background())
	return len(p), nil
}

func (engine *Engine) Close() error {
	return nil
}
