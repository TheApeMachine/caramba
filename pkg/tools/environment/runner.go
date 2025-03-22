package environment

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/containerd/containerd/v2/client"
	"github.com/containerd/containerd/v2/pkg/cio"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Runner struct {
	ctx          context.Context
	cancel       context.CancelFunc
	buffer       *stream.Buffer
	container    *Container
	task         client.Task
	bufIn        *bytes.Buffer
	bufOut       *bytes.Buffer
	bufErr       *bytes.Buffer
	builder      strings.Builder
	promptMarker string
	outputReady  chan struct{}
}

func NewRunner(container *Container) *Runner {
	errnie.Debug("environment.NewRunner")

	var task client.Task
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	runner := &Runner{
		ctx:          ctx,
		cancel:       cancel,
		container:    container,
		task:         task,
		bufIn:        bytes.NewBuffer([]byte{}),
		bufOut:       bytes.NewBuffer([]byte{}),
		bufErr:       bytes.NewBuffer([]byte{}),
		builder:      strings.Builder{},
		promptMarker: "$",
		outputReady:  make(chan struct{}),
	}

	go func() {
		// Create a new FIFO set for the container
		fifos, err := cio.NewFIFOSetInDir("", container.container.ID(), false)

		if errnie.Error(err) != nil {
			return
		}

		// Create the task with empty IO first
		if task, err = container.container.NewTask(ctx, cio.NewCreator()); errnie.Error(err) != nil {
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

	runner.buffer = stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
		errnie.Debug("environment.Runner.buffer.fn")

		runner.bufIn.Reset()
		runner.bufOut.Reset()
		runner.bufErr.Reset()

		params := &provider.Params{}

		if err = artifact.To(params); err != nil {
			return errnie.Error(err)
		}

		runner.bufIn.Write([]byte(params.Messages[len(params.Messages)-1].Content))

		<-runner.outputReady

		runner.builder.Reset()
		runner.builder.WriteString(runner.bufOut.String())
		runner.builder.WriteString(runner.bufErr.String())

		params.Messages = append(params.Messages, &provider.Message{
			Role:    "assistant",
			Content: runner.builder.String(),
		})

		var out []byte
		if out, err = json.Marshal(params); errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		datura.WithPayload(out)(artifact)
		return nil
	})

	var (
		n   int
		err error
	)

	go func() {
		buf := make([]byte, 1024)
		for {
			select {
			case <-runner.ctx.Done():
				return
			default:
				n, err = runner.bufOut.Read(buf)
				if n == 0 || err != nil {
					select {
					case runner.outputReady <- struct{}{}:
					default:
					}
					continue
				}
				// Process any output we did get
				if n > 0 {
					// Keep reading until we get a 0-byte read
					continue
				}
			}
		}
	}()

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
