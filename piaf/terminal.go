package piaf

import (
	"bufio"
	"fmt"
	"os"
	"os/signal" // Added for potential future use if needed here, though likely more in framebuffer
	"syscall"
	"time"

	"golang.org/x/term"
)

// Terminal handles the terminal I/O and raw mode
type Terminal struct {
	editor     *Editor
	oldState   *term.State
	sigChan    chan os.Signal
	shouldQuit bool
	writer     *bufio.Writer
	// Pre-allocated buffer for terminal output
	// outputBuffer []byte
}

// NewTerminal creates a new terminal handler
func NewTerminal(editor *Editor) *Terminal {
	// Use a potentially larger buffer size if needed, bufio default is 4096
	// Increasing buffer size can reduce the number of underlying write syscalls
	// for large updates, but consumes more memory. 8192 is a reasonable starting point.
	writer := bufio.NewWriterSize(os.Stdout, 8192)
	return &Terminal{
		editor:  editor,
		sigChan: make(chan os.Signal, 1),
		writer:  writer,
		// outputBuffer: make([]byte, 0, 4096),
	}
}

// EnableRawMode puts the terminal in raw mode
func (t *Terminal) EnableRawMode() error {
	var err error
	// Use Stdin for raw mode settings
	fd := int(os.Stdin.Fd())
	if !term.IsTerminal(fd) {
		return fmt.Errorf("standard input is not a terminal")
	}
	t.oldState, err = term.MakeRaw(fd)
	if err != nil {
		return fmt.Errorf("could not enable raw mode: %v", err)
	}
	return nil
}

// DisableRawMode restores the terminal to its original state
func (t *Terminal) DisableRawMode() {
	if t.oldState != nil {
		term.Restore(int(os.Stdin.Fd()), t.oldState)
		// Optionally clear screen and show cursor on exit?
		// fmt.Print(clearScreenHome) // Write directly here might be okay for cleanup
		// fmt.Print(showCursor)
	}
}

// flushTerminal flushes the underlying bufio.Writer.
// This should be called once per render frame.
func (t *Terminal) flushTerminal() error {
	return t.writer.Flush()
}

// Define control characters
const (
	ctrlQ = 17 // ASCII value for Ctrl-Q
)

// flushAndLogError is kept for convenience for places outside the main render loop
// where a simple flush + error log is needed (e.g., initial clear, final clear).
func (t *Terminal) flushAndLogError(errMsg string) {
	if err := t.flushTerminal(); err != nil {
		// Writing to stderr might interfere with TUI, consider logging to a file for debugging
		// Or handle errors more gracefully depending on context.
		fmt.Fprintf(os.Stderr, "%s: %v\n", errMsg, err)
	}
}

// Run starts the terminal input loop
func (t *Terminal) Run() error {
	// Set up signal handling
	signal.Notify(t.sigChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGWINCH) // Added SIGWINCH
	defer signal.Stop(t.sigChan)

	// Enable raw mode
	if err := t.EnableRawMode(); err != nil {
		return err
	}
	// Ensure raw mode is disabled even if errors occur later
	defer t.DisableRawMode()

	// Start the editor's background processing goroutine
	t.editor.Run()

	// Initial setup: Clear screen and hide cursor
	// Write directly to the buffered writer
	t.writer.WriteString(hideCursor)
	t.writer.WriteString(clearScreenHome)
	// Flush these initial commands immediately
	if err := t.flushTerminal(); err != nil {
		// Log error, but attempt to continue if possible
		fmt.Fprintf(os.Stderr, "Error flushing buffer on startup: %v\n", err)
	}

	// Trigger initial render (framebuffer needs size info first)
	t.editor.buffer.ForceRedraw() // Ensure the first render is a full one
	if err := t.editor.RenderTo(t.writer); err != nil {
		// Handle initial render error
		fmt.Fprintf(os.Stderr, "Error during initial render: %v\n", err)
		// Consider exiting or trying again
	}
	t.flushAndLogError("Error flushing buffer after initial render")

	// Create input channel
	inputCh := make(chan rune, 10) // Buffer input slightly

	// Input reader goroutine
	go func() {
		// Use a buffered reader for Stdin as well for efficiency
		stdinReader := bufio.NewReader(os.Stdin)
		for {
			r, _, err := stdinReader.ReadRune()
			if err != nil {
				if !t.shouldQuit { // Avoid sending error after quit signal
					// Log error or handle EOF/other issues
					fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
					// Maybe signal quit on read errors?
					t.shouldQuit = true
				}
				close(inputCh)
				return
			}
			// Check for quit command (Ctrl-Q) - maybe make configurable
			if r == ctrlQ {
				t.shouldQuit = true
				close(inputCh)
				return
			}
			// Send rune, non-blocking to avoid deadlock if main loop is busy
			select {
			case inputCh <- r:
			default:
				// Input buffer full, dropping character? Or block?
				// For now, let's block shortly to avoid losing input easily
				// Consider a larger inputCh buffer if this becomes an issue
				select {
				case inputCh <- r:
				case <-time.After(10 * time.Millisecond):
					fmt.Fprintf(os.Stderr, "Input channel full, dropping input.\n")
				}
			}
			if t.shouldQuit { // Check after potential block
				close(inputCh)
				return
			}
		}
	}()

	// Main loop with input handling and rendering
	// Ticker for minimum render interval (~60fps)
	renderTicker := time.NewTicker(time.Millisecond * 16)
	defer renderTicker.Stop()

	for !t.shouldQuit {
		// Prioritize channels
		select {
		case r, ok := <-inputCh:
			if !ok {
				t.shouldQuit = true // Input channel closed, likely intentional quit
				continue
			}
			t.editor.ProcessInput(r)
			// Input implies state changed, so a render is likely needed.
			// Framebuffer dirty flag will confirm if actual content changed.

		case sig := <-t.sigChan:
			switch sig {
			case syscall.SIGINT, syscall.SIGTERM:
				t.shouldQuit = true
				// Optionally signal editor to save or clean up
				// t.editor.commandCh <- CmdQuit
			case syscall.SIGWINCH:
				// Terminal resized, update framebuffer and force full redraw
				t.editor.buffer.HandleResize()
				// Render will happen in the ticker logic below
			}
			if t.shouldQuit {
				continue
			}

		case <-t.editor.ShouldQuit(): // Editor explicitly requested quit (e.g., :q command)
			t.shouldQuit = true
			continue

		case <-renderTicker.C:
			// Render only if editor state is dirty (content changed, mode changed, etc.)
			// The IsDirty check should be comprehensive now.
			if t.editor.buffer.IsDirty() {
				// Pass the buffered writer directly to the editor's RenderTo method
				if err := t.editor.RenderTo(t.writer); err != nil {
					// Handle render error (log, maybe try full redraw next time?)
					fmt.Fprintf(os.Stderr, "Error during render: %v\n", err)
					t.editor.buffer.ForceRedraw() // Force redraw on next attempt
				} else {
					// Single flush after RenderTo has written everything to the buffer
					if err := t.flushTerminal(); err != nil {
						fmt.Fprintf(os.Stderr, "Error flushing terminal: %v\n", err)
						// This is more critical, might indicate terminal issues.
					}
				}
			}
		}
	}

	// Final cleanup before exiting the Run function
	// Ensure cursor is visible and potentially clear screen
	t.writer.WriteString(showCursor)
	t.writer.WriteString(clearScreenHome) // Clear screen fully
	// Final flush
	t.flushAndLogError("Error on final flush")

	return nil // Indicate graceful shutdown
}

// Close cleans up the terminal
func (t *Terminal) Close() error {
	// Disable raw mode happens in defer in Run()
	// Close signal channel? Not strictly necessary as Stop is called.
	// Flush any remaining buffer data (might be unnecessary if Run handles final flush)
	err := t.writer.Flush()
	// Close editor resources
	editorErr := t.editor.Close()
	if err != nil {
		return err // Return flush error first if it occurred
	}
	return editorErr
}
