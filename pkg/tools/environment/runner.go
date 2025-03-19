package environment

import (
	"context"
	"fmt"
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
}

func NewRunner(container *Container) *Runner {
	errnie.Debug("environment.NewRunner")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	runner := &Runner{
		ctx:       ctx,
		cancel:    cancel,
		container: container,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) error {
			errnie.Debug("environment.Runner.buffer.fn")

			task, err := container.container.NewTask(ctx, cio.NewCreator(
				cio.WithStreams(artifact, artifact, artifact),
			))

			if errnie.Error(err) != nil {
				return err
			}

			if err = task.Start(ctx); err != nil {
				return errnie.Error(err)
			}

			errnie.Debug(fmt.Sprintf("Started container task with PID: %d", task.Pid()))

			return nil
		}),
	}

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
