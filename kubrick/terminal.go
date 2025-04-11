package kubrick

import (
	"fmt"
	"io"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/theapemachine/caramba/kubrick/layouts"
	"github.com/theapemachine/caramba/kubrick/types"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// Define control characters
const (
	ctrlQ = 17 // ASCII value for Ctrl-Q
)

// Terminal handles the terminal I/O and raw mode
type Terminal struct {
	*types.Contextualizer

	wg         *sync.WaitGroup
	transport  Transport
	sigChan    chan os.Signal
	shouldQuit bool
	root       layouts.Layout
	err        error
}

type TerminalOption func(*Terminal)

// NewTerminal creates a new terminal handler
func NewTerminal(opts ...TerminalOption) *Terminal {
	terminal := &Terminal{
		Contextualizer: types.NewContextualizer(),
		transport:      NewLocalTransport(),
		sigChan:        make(chan os.Signal, 1),
	}

	for _, opt := range opts {
		opt(terminal)
	}

	terminal.wg.Add(1)

	if err := terminal.render(); err != nil {
		errnie.New(
			errnie.WithType(errnie.RenderError),
			errnie.WithError(err),
		)
		return nil
	}

	return terminal
}

func (terminal *Terminal) render() (err error) {
	if _, ok := terminal.transport.(*LocalTransport); ok {
		signal.Notify(
			terminal.sigChan,
			syscall.SIGINT,
			syscall.SIGTERM,
			syscall.SIGWINCH,
		)

		defer signal.Stop(terminal.sigChan)
	}

	// Enable raw mode
	if err = terminal.transport.SetRawMode(); err != nil {
		terminal.err = errnie.New(
			errnie.WithType(errnie.RenderError),
			errnie.WithError(err),
		)
		return
	}

	defer terminal.transport.RestoreMode()

	// Initial setup
	terminal.transport.Write([]byte(hideCursor))
	terminal.transport.Write([]byte(clearScreenHome))

	// Create input channel
	inputCh := make(chan rune, 10)

	// Input reader goroutine
	go func() {
		defer close(inputCh)

		buf := make([]byte, 1)

		for {
			select {
			case <-terminal.Done():
				terminal.Close()
				return
			default:
				n, err := terminal.transport.Read(buf)
				if err != nil || n == 0 {
					if !terminal.shouldQuit {
						fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
						terminal.shouldQuit = true
					}
					close(inputCh)
					return
				}
				r := rune(buf[0])
				if r == ctrlQ {
					terminal.shouldQuit = true
					close(inputCh)
					return
				}
				select {
				case inputCh <- r:
				default:
					// Drop input if channel is full
				}
			}
		}
	}()

	// Main loop
	displayTicker := time.NewTicker(time.Millisecond * 16)
	defer displayTicker.Stop()

	go func() {
		for {
			select {
			case <-terminal.Done():
				return
			case r, ok := <-inputCh:
				if !ok {
					terminal.shouldQuit = true
					continue
				}
				// Pass input to root if it implements input handling
				if inputHandler, ok := terminal.root.(interface{ HandleInput(rune) }); ok {
					inputHandler.HandleInput(r)
				}

			case sig := <-terminal.sigChan:
				switch sig {
				case syscall.SIGINT, syscall.SIGTERM:
					terminal.shouldQuit = true
				case syscall.SIGWINCH:
					// Terminal resized, update root layout
					if terminal.root != nil {
						width, height, _ := terminal.transport.GetSize()
						terminal.root.SetRect(layouts.Rect{
							Pos: layouts.Position{Row: 0, Col: 0},
							Size: layouts.Size{
								Width:  width,
								Height: height,
							},
						})
					}
				}
				if terminal.shouldQuit {
					continue
				}

			case <-displayTicker.C:
				io.Copy(terminal.transport, terminal.root)
			}
		}
	}()

	return nil
}

func (terminal *Terminal) Read(p []byte) (n int, err error) {
	return terminal.transport.Read(p)
}

func (terminal *Terminal) Write(p []byte) (n int, err error) {
	return terminal.transport.Write(p)
}

func (terminal *Terminal) Close() error {
	terminal.Cancel()

	terminal.transport.Write([]byte(showCursor))
	terminal.transport.Write([]byte(clearScreenHome))

	return terminal.transport.Close()
}

func WithRoot(root layouts.Layout) TerminalOption {
	return func(t *Terminal) {
		t.root = root
	}
}

func WithTransport(transport Transport) TerminalOption {
	return func(t *Terminal) {
		t.transport = transport
	}
}
