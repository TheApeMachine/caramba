package environment

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"syscall"
	"time"

	"github.com/containerd/containerd/v2/client"
	"github.com/containerd/containerd/v2/pkg/cio"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Runner struct {
	ctx       context.Context
	cancel    context.CancelFunc
	buffer    *stream.Buffer
	container *Container
	task      client.Task
	bufIn     *bytes.Buffer
	bufOut    *bytes.Buffer
	bufErr    *bytes.Buffer
	muOutErr  sync.Mutex
	muIn      sync.Mutex
}

func NewRunner(container *Container) *Runner {
	errnie.Debug("environment.NewRunner")

	var task client.Task
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	runner := &Runner{
		ctx:       ctx,
		cancel:    cancel,
		container: container,
		task:      task,
		bufIn:     bytes.NewBuffer([]byte{}),
		bufOut:    bytes.NewBuffer([]byte{}),
		bufErr:    bytes.NewBuffer([]byte{}),
	}

	go func() {
		// Create a new FIFO set for the container
		fifos, err := cio.NewFIFOSetInDir("", container.container.ID(), false)

		if errnie.Error(err) != nil {
			return
		}

		// Create the task with empty IO first
		if task, err = container.container.NewTask(ctx, cio.NewCreator()); err != nil {
			errnie.Error(err)
			return
		}

		// Attach the streams using our containerIO bridge
		attachedFifos, err := cio.NewAttach(cio.WithStreams(
			runner.bufIn,
			io.MultiWriter(runner.bufOut, os.Stdout),
			io.MultiWriter(runner.bufErr, os.Stderr),
		))(fifos)

		if errnie.Error(err) != nil {
			return
		}

		defer attachedFifos.Close()

		if err = task.Start(ctx); errnie.Error(err) != nil {
			return
		}

		errnie.Debug(fmt.Sprintf("Started container task with PID: %d", task.Pid()))

		var status <-chan client.ExitStatus
		if status, err = task.Wait(ctx); errnie.Error(err) != nil {
			return
		}

		stat := <-status
		if stat.ExitCode() != 0 {
			return
		}

	}()

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

		// Get the command from the metadata, which contains all the arguments
		// for the tool call.
		command := datura.GetMetaValue[string](artifact, "command")

		if command == "" {
			return errnie.Error(errors.New("no command"))
		}

		// Write the command to the container.
		runner.muIn.Lock()
		runner.bufIn.Write([]byte(command))
		runner.muIn.Unlock()

		// Wait until the output of the command returns 0 bytes.
		<-waitforoutput()

		runner.muOutErr.Lock()
		datura.WithPayload(
			append(
				runner.bufOut.Bytes(),
				runner.bufErr.Bytes()...,
			),
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

	errnie.Error(runner.task.Kill(runner.ctx, syscall.SIGTERM))
	return runner.buffer.Close()
}
