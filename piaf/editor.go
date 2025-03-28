package piaf

import (
	"bufio"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Editor commands
const (
	CmdMoveLeft = iota
	CmdMoveRight
	CmdMoveUp
	CmdMoveDown
	CmdInsertMode
	CmdNormalMode
	CmdVisualMode
	CmdCommandMode
	CmdDeleteChar
	CmdBackspace
	CmdNewLine
	CmdQuit
	CmdForceQuit
	CmdOpenFile
	CmdSaveFile
	CmdExplorerMode
)

// Special key sequences
const (
	EscapeKey    = 27
	ArrowPrefix  = 91 // '['
	ArrowUp      = 65 // 'A'
	ArrowDown    = 66 // 'B'
	ArrowRight   = 67 // 'C'
	ArrowLeft    = 68 // 'D'
	EnterKey     = 13
	BackspaceKey = 127
)

// Editor manages the editing session
type Editor struct {
	buffer         *Framebuffer
	commandCh      chan int
	inputCh        chan rune
	quitCh         chan struct{} // Channel to signal quit
	escapeSequence bool
	arrowSequence  bool
	commandBuffer  string // Buffer for command mode input
	filename       string // Current file being edited

	// Explorer state
	explorerPath          string        // Current directory path for explorer
	explorerEntries       []os.DirEntry // Entries in the current explorer path
	explorerSelectedIndex int           // Index of the selected entry in explorer
	explorerError         string        // Error message for explorer mode
}

// NewEditor creates a new editor instance
func NewEditor() *Editor {
	return &Editor{
		buffer:         NewFramebuffer(),
		commandCh:      make(chan int, 100),
		inputCh:        make(chan rune, 100),
		quitCh:         make(chan struct{}),
		escapeSequence: false,
		arrowSequence:  false,
		filename:       "", // Initialize filename
		// Initialize explorer state
		explorerPath:          ".", // Start in current directory
		explorerEntries:       nil,
		explorerSelectedIndex: 0,
		explorerError:         "",
	}
}

// ProcessInput handles a keypress and updates the editor state
func (e *Editor) ProcessInput(r rune) {
	// --- Arrow Key Sequence Handling ---
	if e.escapeSequence {
		if e.arrowSequence {
			// Expecting A, B, C, or D
			cmd := -1
			switch r {
			case ArrowUp:
				cmd = CmdMoveUp
			case ArrowDown:
				cmd = CmdMoveDown
			case ArrowRight:
				cmd = CmdMoveRight
			case ArrowLeft:
				cmd = CmdMoveLeft
			}
			if cmd != -1 {
				e.commandCh <- cmd
			}
			// Reset sequence state regardless of valid arrow key
			e.escapeSequence = false
			e.arrowSequence = false
			return
		}
		// Expecting '[' after Escape
		if r == ArrowPrefix {
			e.arrowSequence = true
			// Keep escapeSequence = true, wait for A/B/C/D
			return
		}
		// If we get here, Escape was pressed, then something other than '['
		// Discard the sequence state and process the current rune 'r' normally below.
		e.escapeSequence = false
	}

	// --- Single Escape Key Handling ---
	if r == EscapeKey { // '\x1b'
		currentMode := e.buffer.mode
		if currentMode == ModeInsert || currentMode == ModeVisual || currentMode == ModeCommand {
			// Exit to Normal mode
			if currentMode == ModeInsert && e.buffer.cursor.Col > 0 {
				e.commandCh <- CmdMoveLeft // Move cursor back like vim
			}
			if currentMode == ModeCommand {
				e.commandBuffer = ""        // Clear editor command buffer
				e.buffer.commandBuffer = "" // Clear framebuffer copy too
				e.buffer.dirty = true       // Force redraw to clear command line
			}
			e.commandCh <- CmdNormalMode
			// Reset sequence state just in case
			e.escapeSequence = false
			e.arrowSequence = false
		} else if currentMode == ModeNormal {
			// If already in normal mode, maybe start an escape sequence?
			// Or just ignore plain escape? Vim ignores plain escape in normal mode. Let's ignore.
			// If we wanted arrow keys, we'd set e.escapeSequence = true here.
			// For now, let's require Esc+[ for arrows.
			e.escapeSequence = true // Set flag to check for '[' next time
		}
		return // Handled the Escape key itself
	}

	// Normal input processing
	switch e.buffer.mode {
	case ModeNormal:
		e.handleNormalMode(r)
	case ModeInsert:
		e.handleInsertMode(r)
	case ModeVisual:
		e.handleVisualMode(r)
	case ModeCommand:
		e.handleCommandMode(r)
	case ModeExplorer:
		e.handleExplorerMode(r)
	default:
		// Log or handle unexpected mode? Reset to normal?
		// fmt.Fprintf(os.Stderr, "WARN: Unexpected mode %d encountered in ProcessInput. Resetting to Normal.\n", e.buffer.mode)
		e.commandCh <- CmdNormalMode
	}
}

// handleNormalMode processes input in normal mode
func (e *Editor) handleNormalMode(r rune) {
	switch r {
	case 'h':
		e.commandCh <- CmdMoveLeft
	case 'l':
		e.commandCh <- CmdMoveRight
	case 'k':
		e.commandCh <- CmdMoveUp
	case 'j':
		e.commandCh <- CmdMoveDown
	case 'i':
		e.commandCh <- CmdInsertMode
	case 'v':
		e.commandCh <- CmdVisualMode
	case 'x':
		e.commandCh <- CmdDeleteChar
	case ':':
		e.commandBuffer = ":"        // Still need to prime the buffer
		e.buffer.commandBuffer = ":" // Update framebuffer copy too
		e.buffer.dirty = true
		e.commandCh <- CmdCommandMode // Send command to switch mode
	}
}

// handleInsertMode processes input in insert mode
func (e *Editor) handleInsertMode(r rune) {
	switch r {
	case EscapeKey:
		// Don't handle escape here, it's now fully handled in ProcessInput
		// to properly manage escape sequences
	case EnterKey:
		e.commandCh <- CmdNewLine
	case BackspaceKey:
		e.commandCh <- CmdBackspace
	default:
		e.inputCh <- r
	}
}

// handleVisualMode processes input in visual mode
func (e *Editor) handleVisualMode(r rune) {
	switch r {
	case EscapeKey:
		e.commandCh <- CmdNormalMode
	case 'h':
		e.commandCh <- CmdMoveLeft
	case 'l':
		e.commandCh <- CmdMoveRight
	case 'k':
		e.commandCh <- CmdMoveUp
	case 'j':
		e.commandCh <- CmdMoveDown
	}
}

// handleCommandMode processes input in command mode
func (e *Editor) handleCommandMode(r rune) {
	switch r {
	case EnterKey:
		e.executeCommand()
		e.commandBuffer = ""
		e.buffer.SetMode(ModeNormal)
	case BackspaceKey:
		if len(e.commandBuffer) > 1 { // Keep the colon
			e.commandBuffer = e.commandBuffer[:len(e.commandBuffer)-1]
			e.buffer.commandBuffer = e.commandBuffer // Update framebuffer copy
			e.buffer.dirty = true                    // Mark buffer as dirty
		}
	default:
		e.commandBuffer += string(r)
		e.buffer.commandBuffer = e.commandBuffer // Update framebuffer copy
		e.buffer.dirty = true                    // Mark buffer as dirty
	}
}

// executeCommand executes the command in the command buffer
func (e *Editor) executeCommand() {
	cmdLine := strings.TrimSpace(e.commandBuffer[1:]) // Remove the leading colon
	e.buffer.commandBuffer = ""                       // Clear framebuffer's command buffer copy
	parts := strings.Fields(cmdLine)                  // Split command and arguments
	if len(parts) == 0 {
		return // Empty command
	}
	command := parts[0]
	args := parts[1:] // Arguments for the command

	switch command {
	case "q", "quit":
		// TODO: Add check for unsaved changes
		e.commandCh <- CmdQuit
	case "q!": // Force quit
		e.commandCh <- CmdForceQuit
	case "w", "write":
		filename := e.filename
		if len(args) > 0 {
			filename = args[0] // Save As
		}
		if filename == "" {
			e.buffer.SetStatusMessage("No file name") // Need to display this somehow
			return
		}
		e.filename = filename // Update current filename if saved to a new one
		e.commandCh <- CmdSaveFile
	case "o", "open", "e", "edit": // Treat o, open, e, edit similarly
		if len(args) == 0 {
			e.buffer.SetStatusMessage("File name missing") // Need status message display
			return
		}
		e.filename = args[0]
		e.commandCh <- CmdOpenFile
	case "E", "Ex": // Command to open explorer
		path := "." // Default to current directory
		if len(args) > 0 {
			path = args[0] // Allow specifying a starting path
		}
		e.explorerPath = path
		if err := e.readExplorerDir(); err != nil {
			// Handle error reading directory - maybe show in status?
			e.buffer.SetStatusMessage("Error reading directory: " + err.Error())
			// Stay in command mode or normal mode if dir read fails?
			// Let's stay in normal mode for now.
			e.commandCh <- CmdNormalMode
		} else {
			e.commandCh <- CmdExplorerMode // Switch to explorer mode
		}
	// Add more commands here as needed
	default:
		// Optionally display an error message for unknown commands
		e.buffer.SetStatusMessage("Unknown command: " + command) // Need status message display
	}
}

// ProcessCommand executes an editor command
func (e *Editor) ProcessCommand(cmd int) {
	switch cmd {
	case CmdMoveLeft:
		e.buffer.MoveCursor(e.buffer.cursor.Row, e.buffer.cursor.Col-1)
	case CmdMoveRight:
		e.buffer.MoveCursor(e.buffer.cursor.Row, e.buffer.cursor.Col+1)
	case CmdMoveUp:
		e.buffer.MoveCursor(e.buffer.cursor.Row-1, e.buffer.cursor.Col)
	case CmdMoveDown:
		e.buffer.MoveCursor(e.buffer.cursor.Row+1, e.buffer.cursor.Col)
	case CmdInsertMode:
		e.buffer.SetMode(ModeInsert)
	case CmdNormalMode:
		e.buffer.SetMode(ModeNormal)
	case CmdVisualMode:
		e.buffer.SetMode(ModeVisual)
	case CmdCommandMode:
		e.buffer.SetMode(ModeCommand)
	case CmdDeleteChar:
		e.buffer.DeleteRune()
	case CmdBackspace:
		e.buffer.DeleteRune()
	case CmdNewLine:
		e.buffer.InsertNewLine()
	case CmdQuit:
		// TODO: Check for unsaved changes before quitting
		close(e.quitCh)
	case CmdForceQuit:
		close(e.quitCh) // Quit unconditionally
	case CmdOpenFile:
		content, err := os.ReadFile(e.filename) // Use os.ReadFile
		if err != nil {
			// Handle error (e.g., file not found) - display message?
			// If file doesn't exist, should probably start with empty buffer
			if os.IsNotExist(err) {
				e.buffer.Reset()        // Start fresh if file doesn't exist
				e.buffer.SetContent("") // Ensure it's really empty
				e.buffer.SetStatusMessage("New file: " + e.filename)
			} else {
				e.buffer.SetStatusMessage("Error opening file: " + err.Error())
			}
		} else {
			e.buffer.SetContent(string(content))
			e.buffer.SetStatusMessage("Opened: " + e.filename)
		}
		// Reset view and cursor after loading/creating
		e.buffer.viewOffset = Position{0, 0}
		e.buffer.MoveCursor(0, 0)
	case CmdSaveFile:
		content := e.buffer.GetContent()
		// Use 0644 permissions for standard text files
		err := os.WriteFile(e.filename, []byte(content), 0644)
		if err != nil {
			e.buffer.SetStatusMessage("Error saving file: " + err.Error())
		} else {
			e.buffer.SetStatusMessage("Saved: " + e.filename)
			// TODO: Mark buffer as not dirty after saving
		}
	case CmdExplorerMode:
		e.buffer.SetMode(ModeExplorer)
		// Need to tell the buffer about explorer data for rendering
		e.buffer.SetExplorerData(e.explorerPath, e.explorerEntries, e.explorerSelectedIndex, e.explorerError)
		e.buffer.ForceRedraw() // Force redraw for the new view
	}
}

// Run starts the editor main loop
func (e *Editor) Run() {
	go func() {
		for {
			select {
			case cmd := <-e.commandCh:
				e.ProcessCommand(cmd)
			case r := <-e.inputCh:
				if e.buffer.mode == ModeInsert {
					e.buffer.InsertRune(r)
				}
			case <-e.quitCh:
				return
			}
		}
	}()
}

// RenderTo writes the current editor view directly to the provided writer.
func (e *Editor) RenderTo(writer *bufio.Writer) error {
	// Delegate rendering to the framebuffer, passing the writer.
	return e.buffer.RenderTo(writer)
}

// Read implements io.Reader
// It now reconstructs the content from the framebuffer's lines.
func (e *Editor) Read(p []byte) (n int, err error) {
	content := e.buffer.GetContent() // Get content directly from buffer
	if len(content) == 0 {
		// This might happen if the buffer is truly empty. Decide if EOF is correct.
		// If it should wait for content, different logic is needed.
		// For now, assume empty content means EOF for a single read.
		return 0, io.EOF
	}
	n = copy(p, []byte(content))
	// Decide if Read should consume the content or always return the full buffer?
	// Standard io.Reader expects consumption, but for a TUI snapshot, maybe not?
	// Current implementation returns the full content on each call until p is full.
	// If subsequent calls should return remaining data, state needs to be tracked.
	// For now, let's assume it provides a snapshot and might return less than EOF
	// if p is smaller than content.
	if n < len(content) {
		// Indicate more data might be available on subsequent calls if Read contract requires it.
		// However, without tracking read state, this simple copy is snapshot-like.
		err = nil // Don't signal EOF if buffer wasn't fully copied to p
	} else {
		// If we copied everything, signal EOF *for this snapshot*.
		err = io.EOF
	}
	return n, err
}

// Write implements io.Writer
func (e *Editor) Write(p []byte) (n int, err error) {
	e.buffer.SetContent(string(p))
	// Reset view and cursor after setting content externally
	e.buffer.viewOffset = Position{0, 0}
	e.buffer.MoveCursor(0, 0) // Move cursor to beginning
	e.filename = ""           // Clear filename when content is set externally via Write
	return len(p), nil
}

// Close implements io.Closer
func (e *Editor) Close() error {
	close(e.commandCh)
	close(e.inputCh)
	return nil
}

// ShouldQuit returns the quit channel for monitoring exit condition
func (e *Editor) ShouldQuit() <-chan struct{} {
	return e.quitCh
}

// handleExplorerMode processes input in explorer mode
func (e *Editor) handleExplorerMode(r rune) {
	switch r {
	case 'j', ArrowDown:
		if e.explorerSelectedIndex < len(e.explorerEntries)-1 {
			e.explorerSelectedIndex++
			e.buffer.SetExplorerData(e.explorerPath, e.explorerEntries, e.explorerSelectedIndex, e.explorerError) // Update selection
		}
	case 'k', ArrowUp:
		if e.explorerSelectedIndex > 0 {
			e.explorerSelectedIndex--
			e.buffer.SetExplorerData(e.explorerPath, e.explorerEntries, e.explorerSelectedIndex, e.explorerError) // Update selection
		}
	case EnterKey: // Select file or directory
		if e.explorerSelectedIndex < 0 || e.explorerSelectedIndex >= len(e.explorerEntries) {
			break // No valid selection
		}
		selectedEntry := e.explorerEntries[e.explorerSelectedIndex]
		newPath := filepath.Join(e.explorerPath, selectedEntry.Name())

		if selectedEntry.IsDir() {
			// Navigate into directory
			e.explorerPath = newPath
			if err := e.readExplorerDir(); err != nil {
				// Error reading new dir, stay in current dir but show error
				e.explorerError = "Error: " + err.Error() // Keep the old path, but show error
				// Need to reset path back if read failed?
				// Let's try just updating the buffer with the error
				e.buffer.SetExplorerData(e.explorerPath, e.explorerEntries, e.explorerSelectedIndex, e.explorerError)
			} else {
				// Successfully read new directory, update buffer
				e.buffer.SetExplorerData(e.explorerPath, e.explorerEntries, e.explorerSelectedIndex, e.explorerError)
			}
		} else {
			// Open the file
			e.filename = newPath
			e.commandCh <- CmdOpenFile   // Use existing command to load file
			e.commandCh <- CmdNormalMode // Switch back to normal mode
			e.buffer.ClearExplorerData() // Tell buffer to hide explorer
			// ForceRedraw is handled by CmdOpenFile/CmdNormalMode via SetMode/SetContent
		}
	case EscapeKey, 'q': // Exit explorer mode
		e.commandCh <- CmdNormalMode
		// Need to tell buffer to stop showing explorer
		e.buffer.ClearExplorerData()
		e.buffer.ForceRedraw()
	default:
		// Ignore other keys for now
	}
}

// readExplorerDir reads the directory content for the explorer view
func (e *Editor) readExplorerDir() error {
	e.explorerError = "" // Clear previous error
	entries, err := os.ReadDir(e.explorerPath)
	if err != nil {
		e.explorerError = err.Error()
		e.explorerEntries = nil // Clear entries on error
		return err
	}
	e.explorerEntries = entries
	e.explorerSelectedIndex = 0 // Reset selection to the top
	return nil
}
