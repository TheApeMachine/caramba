package environment

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Runner manages command execution and I/O operations within a container environment.

It provides a buffered interface for running commands and handling interactive
input/output with a container. The Runner maintains separate buffers for stdin,
stdout, and stderr streams, and uses mutexes to ensure thread-safe operations.
*/
type Runner struct {
	ctx      context.Context
	cancel   context.CancelFunc
	runtime  Runtime
	bufIn    *bytes.Buffer
	bufOut   *bytes.Buffer
	bufErr   *bytes.Buffer
	muOutErr sync.Mutex
	muIn     sync.Mutex
}

/*
NewRunner creates a new Runner instance with the specified runtime.

It initializes the runner with a 5-minute timeout context and sets up buffered
I/O streams for container interaction. The runtime's I/O is attached to the
runner's buffers for command execution and interactive operations.

Returns nil if I/O attachment fails.
*/
func NewRunner(runtime Runtime) *Runner {
	errnie.Debug("environment.NewRunner")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)

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
		errnie.New(errnie.WithError(err))
		return nil
	}

	return runner
}

func (runner *Runner) Execute(
	buffer chan map[string]any,
	command string,
	input string,
) chan map[string]any {
	out := make(chan map[string]any)

	// Detect when the command has finished executing and the
	// output has been written to the buffer.
	waitforoutput := func() chan []byte {
		var (
			n           int
			err         error
			outputCh    = make(chan []byte, 1)
			noDataCount = 0
			output      = bytes.NewBuffer([]byte{})
		)

		go func() {
			buf := make([]byte, 1024)
			for {
				select {
				case <-runner.ctx.Done():
					outputCh <- output.Bytes()
					close(outputCh)
					return
				default:
					runner.muOutErr.Lock()
					n, err = runner.bufOut.Read(buf)
					runner.muOutErr.Unlock()

					if err != nil && err != io.EOF {
						errnie.New(errnie.WithError(err))
						outputCh <- output.Bytes()
						close(outputCh)
						return
					}

					if n > 0 {
						output.Write(buf[:n])
						errnie.Debug("environment.Runner.waitforoutput.n", "n", n, "buf", string(buf[:n]))
						noDataCount = 0
						continue
					}

					// Only consider command complete after multiple zero reads
					noDataCount++
					if noDataCount > 5 {
						outputCh <- output.Bytes()
						close(outputCh)
						return
					}

					time.Sleep(100 * time.Millisecond)
				}
			}
		}()

		return outputCh
	}

	go func() {
		defer close(out)

		artifact := <-buffer

		if command == "" && input == "" {
			errnie.New(errnie.WithError(errors.New("no command or input")))
			return
		}

		// Handle input differently from commands
		if input != "" {
			errnie.Debug("environment.Runner.buffer.fn.input", "input", input)

			// For input, just write to stdin and wait for output
			if !strings.HasSuffix(input, "\n") {
				input = input + "\n"
			}

			runner.muIn.Lock()
			runner.bufIn.Write([]byte(input))
			runner.muIn.Unlock()

			// Wait for the program's response
			output := <-waitforoutput()

			errnie.Debug("environment.Runner.buffer.fn.out", "out", string(output))
			artifact["output"] = string(output)

			out <- artifact
		}

		// For commands, reset all buffers
		runner.muIn.Lock()
		runner.bufIn.Reset()
		runner.muIn.Unlock()

		runner.muOutErr.Lock()
		runner.bufOut.Reset()
		runner.bufErr.Reset()
		runner.muOutErr.Unlock()

		// Handle command execution
		runner.muIn.Lock()
		errnie.Debug("environment.Runner.buffer.fn.command", "command", command)
		runner.bufIn.Write([]byte(command))
		runner.muIn.Unlock()

		if err := runner.runtime.ExecuteCommand(runner.ctx, command, runner.bufOut, runner.bufErr); err != nil {
			errnie.New(errnie.WithError(err))
			return
		}

		// Wait until the output of the command returns 0 bytes
		output := <-waitforoutput()

		runner.muOutErr.Lock()
		if bytes.Contains(runner.bufErr.Bytes(), []byte("EOFError: EOF when reading a line")) {
			// If this is an EOF during input read, add explicit prompt
			output = append(output, []byte("\n[WAITING FOR INPUT] Please provide input for the program.")...)
		}

		if len(output) == 0 {
			output = []byte("Command executed successfully")
		}

		errnie.Debug("environment.Runner.buffer.fn.out", "out", string(output))
		artifact["output"] = string(output)
		runner.muOutErr.Unlock()

		out <- artifact
	}()

	return out
}

/*
ExecuteCommand runs a command in the container and returns the result.
This is a wrapper around the runtime's ExecuteCommand method that handles
the buffer management internally.
*/
func (runner *Runner) ExecuteCommand(ctx context.Context, command string, stdout, stderr io.Writer) error {
	return runner.runtime.ExecuteCommand(ctx, command, stdout, stderr)
}

/*
SendInput sends input to the container and returns the resulting output.
This method handles locking and buffer management internally.
*/
func (runner *Runner) SendInput(input string) (string, error) {
	// Write to the input buffer
	runner.muIn.Lock()
	_, err := runner.bufIn.Write([]byte(input))
	runner.muIn.Unlock()

	if err != nil {
		return "", err
	}

	// Give time for the process to respond
	time.Sleep(500 * time.Millisecond)

	// Read from the output buffer
	runner.muOutErr.Lock()
	outputBytes := make([]byte, runner.bufOut.Len())
	_, err = runner.bufOut.Read(outputBytes)
	runner.bufOut.Reset() // Clear the buffer after reading
	runner.muOutErr.Unlock()

	if err != nil && err != io.EOF {
		return "", err
	}

	return string(outputBytes), nil
}
