package spinner

import (
	"container/ring"
	"context"
	"sync"
	"time"

	"github.com/theapemachine/caramba/kubrick/types"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var (
	defaultFrames = []rune{
		'⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏',
	}

	successFrame = '✓'
	failureFrame = '✗'
)

type Spinner struct {
	wg       *sync.WaitGroup
	ctx      context.Context
	cancel   context.CancelFunc
	frames   *ring.Ring
	label    string
	artifact *datura.Artifact
	state    types.State
	err      error
}

type SpinnerOption func(*Spinner)

func NewSpinner(options ...SpinnerOption) *Spinner {
	frames := ring.New(len(defaultFrames))

	for _, frame := range defaultFrames {
		frames.Value = frame
		frames = frames.Next()
	}

	spinner := &Spinner{
		wg:       &sync.WaitGroup{},
		frames:   frames,
		artifact: datura.New(),
		state:    types.StateCreated,
	}

	for _, option := range options {
		option(spinner)
	}

	spinner.wg.Add(1)

	if err := spinner.render(); err != nil {
		spinner.err = err
	}

	return spinner
}

// Read implements io.Reader - streams the rendered view
func (spinner *Spinner) Read(p []byte) (n int, err error) {
	if n, spinner.err = spinner.artifact.Read(p); spinner.err != nil {
		spinner.state = types.StateErrored
		return n, errnie.Error(spinner.err)
	}

	return n, spinner.err
}

// Write implements io.Writer - updates spinner state based on commands
func (spinner *Spinner) Write(p []byte) (n int, err error) {
	if n, spinner.err = spinner.artifact.Write(p); spinner.err != nil {
		spinner.state = types.StateErrored
		return n, errnie.Error(spinner.err)
	}

	return n, spinner.err
}

// Close implements io.Closer
func (spinner *Spinner) Close() error {
	switch spinner.state {
	case types.StateCanceled:
		return spinner.artifact.Close()
	case types.StateRunning:
		spinner.cancel()
		return spinner.artifact.Close()
	case types.StateErrored:
		return spinner.artifact.Close()
	case types.StateClosed:
		return spinner.err
	case types.StateSuccess, types.StateFailure:
		return spinner.artifact.Close()
	}

	return spinner.err
}

func (spinner *Spinner) render() (err error) {
	spinner.state = types.StateRunning

	go func() {
		spinner.wg.Wait()

		for {
			select {
			case <-spinner.ctx.Done():
				// We cancelled ourselves.
				spinner.state = types.StateCanceled
				spinner.Close()
				return
			case <-time.After(100 * time.Millisecond):
				switch spinner.state {
				case types.StateRunning:
					spinner.artifact.Write([]byte{byte(spinner.frames.Value.(rune))})
					spinner.frames.Next()
				case types.StateUpdated:
					status := datura.GetMetaValue[string](spinner.artifact, "status")

					switch status {
					case "success":
						spinner.state = types.StateSuccess
						datura.WithPayload([]byte{byte(successFrame)})(spinner.artifact)
					case "failure":
						spinner.state = types.StateFailure
						datura.WithPayload([]byte{byte(failureFrame)})(spinner.artifact)
					}

					label := datura.GetMetaValue[string](spinner.artifact, "label")

					if label != "" {
						spinner.label = label
					}
				}
			}
		}
	}()

	return nil
}

func (spinner *Spinner) WithContext(ctx context.Context) {
	spinner.ctx, spinner.cancel = context.WithCancel(ctx)
	spinner.wg.Done()
}

func WithLabel(label string) SpinnerOption {
	return func(s *Spinner) {
		s.label = label
	}
}
