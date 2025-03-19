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

	var task client.Task
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	runner := &Runner{
		ctx:       ctx,
		cancel:    cancel,
		container: container,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("environment.Runner.buffer.fn")

			if task, err = container.container.NewTask(ctx, cio.NewCreator(
				cio.WithStreams(artifact, artifact, artifact),
			)); err != nil {
				return errnie.Error(err)
			}

			if err = task.Start(ctx); err != nil {
				return errnie.Error(err)
			}

			errnie.Debug(fmt.Sprintf("Started container task with PID: %d", task.Pid()))

			var status <-chan client.ExitStatus

			if status, err = task.Wait(ctx); err != nil {
				return errnie.Error(err)
			}

			stat := <-status

			if stat.ExitCode() != 0 {
				return errnie.Error(fmt.Errorf("container exited with code %d", stat.ExitCode()))
			}

			return nil
		}),
		task: task,
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
