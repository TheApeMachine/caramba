package kubrick

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/theapemachine/caramba/kubrick/layouts"
)

// Define control characters
const (
	ctrlQ = 17 // ASCII value for Ctrl-Q
)

// Terminal handles the terminal I/O and raw mode
type Terminal struct {
	pctx       context.Context
	ctx        context.Context
	cancel     context.CancelFunc
	transport  Transport
	sigChan    chan os.Signal
	shouldQuit bool
	root       layouts.Layout
	err        error
}

type TerminalOption func(*Terminal)

// NewTerminal creates a new terminal handler
func NewTerminal(opts ...TerminalOption) *Terminal {
	ctx, cancel := context.WithCancel(context.Background())

	terminal := &Terminal{
		ctx:       ctx,
		cancel:    cancel,
		transport: NewLocalTransport(),
		sigChan:   make(chan os.Signal, 1),
	}

	for _, opt := range opts {
		opt(terminal)
	}

	go terminal.run()
	return terminal
}

// run starts the terminal input and display loop
func (terminal *Terminal) run() {
	// Set up signal handling for local terminals
	if _, ok := terminal.transport.(*LocalTransport); ok {
		signal.Notify(terminal.sigChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGWINCH)
		defer signal.Stop(terminal.sigChan)
	}

	// Enable raw mode
	if err := terminal.transport.SetRawMode(); err != nil {
		terminal.err = fmt.Errorf("error setting raw mode: %v", err)
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
		for !terminal.shouldQuit {
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
	}()

	// Main loop
	displayTicker := time.NewTicker(time.Millisecond * 16)
	defer displayTicker.Stop()

	for !terminal.shouldQuit {
		select {
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

	// Final cleanup
	terminal.transport.Write([]byte(showCursor))
	terminal.transport.Write([]byte(clearScreenHome))
}

func (terminal *Terminal) Read(p []byte) (n int, err error) {
	return terminal.transport.Read(p)
}

func (terminal *Terminal) Write(p []byte) (n int, err error) {
	return terminal.transport.Write(p)
}

func (terminal *Terminal) Close() error {
	terminal.cancel()
	terminal.shouldQuit = true
	return terminal.transport.Close()
}

func WithRoot(root layouts.Layout) TerminalOption {
	return func(t *Terminal) {
		t.root = root
		if ctxAware, ok := root.(layouts.ContextAware); ok {
			ctxAware.WithContext(t.ctx)
		}
	}
}

func WithTransport(transport Transport) TerminalOption {
	return func(t *Terminal) {
		t.transport = transport
	}
}

func WithContext(ctx context.Context) TerminalOption {
	return func(t *Terminal) {
		t.pctx = ctx
	}
}

// SetRoot changes the terminal's root layout
func (terminal *Terminal) SetRoot(root layouts.Layout) {
	if ctxAware, ok := root.(layouts.ContextAware); ok {
		ctxAware.WithContext(terminal.ctx)
	}
	terminal.root = root
}
