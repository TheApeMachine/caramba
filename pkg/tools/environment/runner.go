package environment

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Runner struct {
	ctx      context.Context
	cancel   context.CancelFunc
	buffer   *stream.Buffer
	runtime  Runtime
	bufIn    *bytes.Buffer
	bufOut   *bytes.Buffer
	bufErr   *bytes.Buffer
	muOutErr sync.Mutex
	muIn     sync.Mutex
}

func NewRunner(runtime Runtime) *Runner {
	errnie.Debug("environment.NewRunner")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	runner := &Runner{
		ctx:     ctx,
		cancel:  cancel,
		runtime: runtime,
		bufIn:   bytes.NewBuffer([]byte{}),
		bufOut:  bytes.NewBuffer([]byte{}),
		bufErr:  bytes.NewBuffer([]byte{}),
	}

	// Attach IO
	if err := runtime.AttachIO(runner.bufIn, runner.bufOut, runner.bufErr); err != nil {
		errnie.Error(err)
		return nil
	}

	// Detect when the command has finished executing and the
	// output has been written to the buffer.
	waitforoutput := func() chan struct{} {
		var (
			n   int
			err error
			ch  = make(chan struct{})
		)

		go func() {
			buf := make([]byte, 1024)
			for {
				select {
				case <-runner.ctx.Done():
					return
				default:
					runner.muOutErr.Lock()
					n, err = runner.bufOut.Read(buf)
					runner.muOutErr.Unlock()

					if n == 0 || err != nil {
						select {
						case ch <- struct{}{}:
							return
						default:
						}
						continue
					}
					if n > 0 {
						continue
					}
				}
			}
		}()

		return ch
	}

	runner.buffer = stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
		errnie.Debug("environment.Runner.buffer.fn")

		runner.muIn.Lock()
		runner.bufIn.Reset()
		runner.muIn.Unlock()

		runner.muOutErr.Lock()
		runner.bufOut.Reset()
		runner.bufErr.Reset()
		runner.muOutErr.Unlock()

		// Get the command from the metadata
		command := datura.GetMetaValue[string](artifact, "command")
		if command == "" {
			return errnie.Error(errors.New("no command"))
		}

		// Write the command and execute it
		runner.muIn.Lock()
		errnie.Debug("environment.Runner.buffer.fn.command", "command", command)
		runner.bufIn.Write([]byte(command))
		runner.muIn.Unlock()

		if err := runner.runtime.ExecuteCommand(runner.ctx, command); err != nil {
			return errnie.Error(fmt.Errorf("failed to execute command: %w", err))
		}

		// Wait until the output of the command returns 0 bytes
		<-waitforoutput()

		runner.muOutErr.Lock()
		out := append(
			runner.bufOut.Bytes(),
			runner.bufErr.Bytes()...,
		)

		if len(out) == 0 {
			out = []byte("Command executed successfully")
		}

		errnie.Debug("environment.Runner.buffer.fn.out", "out", string(out))

		datura.WithPayload(
			out,
		)(artifact)
		runner.muOutErr.Unlock()

		return nil
	})

	return runner
}

func (runner *Runner) Read(p []byte) (n int, err error) {
	errnie.Debug("environment.Runner.Read")
	return runner.buffer.Read(p)
}

func (runner *Runner) Write(p []byte) (n int, err error) {
	errnie.Debug("environment.Runner.Write")
	return runner.buffer.Write(p)
}

func (runner *Runner) Close() error {
	errnie.Debug("environment.Runner.Close")
	runner.cancel()

	if err := runner.runtime.StopContainer(runner.ctx); err != nil {
		errnie.Error(fmt.Errorf("failed to stop container: %w", err))
	}

	return runner.buffer.Close()
}
