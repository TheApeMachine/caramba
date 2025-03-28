package piaf

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"golang.org/x/term"
)

const (
	clearScreenHome = "\033[2J\033[H"
	clearLine       = "\033[K"
	hideCursor      = "\033[?25l"
	showCursor      = "\033[?25h"
	redText         = "\033[31m"
	grayText        = "\033[90m"
	resetAttrs      = "\033[0m"
	reverseVideo    = "\033[7m"

	// Vim-like modes
	ModeNormal = iota
	ModeInsert
	ModeVisual
	ModeCommand
	ModeExplorer

	// Minimum dimensions
	minWidth  = 40
	minHeight = 10

	// Status message display
	statusMessageTimeout = 5 * time.Second // Duration to display status messages
)

// Framebuffer manages the text content and rendering
type Framebuffer struct {
	// buffer *bytes.Buffer // REMOVED
	// renderBuf     *bytes.Buffer // REMOVED
	lines         []string // Content split into lines for easier manipulation
	cursor        Position // Current cursor position
	mode          int      // Current editor mode
	dirty         bool     // Whether buffer needs rendering (content, cursor, mode, view, size change)
	viewOffset    Position // Scroll position for viewport
	commandBuffer string   // Buffer for command mode

	height    int
	heightStr string // Keep string version for frequent use in ANSI codes
	width     int
	widthStr  string // Keep string version

	// Fields for differential rendering
	prevLines      []string // Lines rendered in the previous frame
	prevCursor     Position // Cursor position in the previous frame
	prevMode       int      // Mode in the previous frame
	prevViewOffset Position // View offset in the previous frame
	prevWidth      int      // Terminal width in the previous frame
	prevHeight     int      // Terminal height in the previous frame
	forceRedraw    bool     // Flag to force a full redraw

	// Reusable buffer for number conversions to avoid allocs
	numBuf []byte

	// Status message display
	statusMessage         string    // Message to display in the footer
	statusMessageTime     time.Time // Time the message was set
	prevStatusMessage     string    // Previous status message
	prevStatusMessageTime time.Time // Previous status message time

	// Explorer state for rendering
	explorerVisible           bool          // Is the explorer view active?
	explorerPath              string        // Path being viewed
	explorerEntries           []os.DirEntry // Directory entries to display
	explorerSelectedIndex     int           // Index of the selected item
	explorerError             string        // Error message (if any)
	prevExplorerVisible       bool          // Previous explorer visibility
	prevExplorerPath          string        // Previous explorer path
	prevExplorerEntries       []os.DirEntry // Previous explorer entries (might not need full copy?)
	prevExplorerSelectedIndex int           // Previous selected index
	prevExplorerError         string        // Previous error message
}

// Position represents 2D coordinates in the buffer
type Position struct {
	Row int
	Col int
}

// NewFramebuffer creates a new editor buffer
func NewFramebuffer() *Framebuffer {
	fb := &Framebuffer{
		lines:      []string{""}, // Start with one empty line
		cursor:     Position{0, 0},
		mode:       ModeNormal,
		dirty:      true, // Initially dirty to force first render
		viewOffset: Position{0, 0},
		// Initialize previous state - ensure first render is a full redraw
		forceRedraw: true,
		prevMode:    -1,                  // Invalid mode to ensure header redraws on first render
		numBuf:      make([]byte, 0, 16), // Small buffer for strconv.AppendInt
	}

	// Get initial size
	_ = fb.updateTerminalSize() // Ignore error on initial setup, use defaults if fails

	return fb
}

// updateTerminalSize gets the current terminal dimensions and updates relevant fields.
// Returns true if the size changed.
func (f *Framebuffer) updateTerminalSize() bool {
	width, height, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil || width < minWidth || height < minHeight {
		// Fallback to minimum dimensions if we can't get size or if terminal is too small
		width = minWidth
		height = minHeight
	}

	if width != f.width || height != f.height {
		f.width = width
		f.height = height
		// Pre-calculate string versions for efficiency in rendering loops
		f.numBuf = f.numBuf[:0] // Clear buffer before reuse
		f.widthStr = string(strconv.AppendInt(f.numBuf, int64(f.width), 10))
		f.numBuf = f.numBuf[:0]
		f.heightStr = string(strconv.AppendInt(f.numBuf, int64(f.height), 10))
		return true
	}
	return false
}

// HandleResize should be called when terminal size changes (e.g., on SIGWINCH).
func (f *Framebuffer) HandleResize() {
	if f.updateTerminalSize() {
		f.forceRedraw = true // Force a full redraw if size changed
		f.dirty = true       // Mark as dirty
	}
}

// ForceRedraw marks the buffer to require a full redraw on the next render cycle.
func (f *Framebuffer) ForceRedraw() {
	f.forceRedraw = true
	f.dirty = true // ForceRedraw implies dirty
}

// Reset clears the buffer content
func (f *Framebuffer) Reset() {
	// f.buffer.Reset() REMOVED
	// f.renderBuf.Reset() REMOVED
	f.lines = []string{""} // Reset to a single empty line
	f.cursor = Position{0, 0}
	f.viewOffset = Position{0, 0}
	f.commandBuffer = ""
	f.dirty = true
	f.forceRedraw = true // Force redraw after reset
}

// InsertRune inserts a character at the current cursor position
func (f *Framebuffer) InsertRune(r rune) {
	// Ensure cursor row is valid
	if f.cursor.Row < 0 {
		f.cursor.Row = 0
	}
	if f.cursor.Row >= len(f.lines) {
		// If cursor is beyond existing lines (shouldn't normally happen with clamping),
		// add lines up to the cursor row.
		for i := len(f.lines); i <= f.cursor.Row; i++ {
			f.lines = append(f.lines, "")
		}
	}

	line := f.lines[f.cursor.Row]

	// Clamp column before insertion
	if f.cursor.Col < 0 {
		f.cursor.Col = 0
	}
	if f.cursor.Col > len(line) {
		f.cursor.Col = len(line)
	}

	// Insert rune using string slicing (efficient for typical line lengths)
	newLine := line[:f.cursor.Col] + string(r) + line[f.cursor.Col:]
	f.lines[f.cursor.Row] = newLine
	f.cursor.Col++
	f.dirty = true
	// No need to adjust view offset here, done by MoveCursor if needed
}

// InsertNewLine inserts a newline at the current cursor position
func (f *Framebuffer) InsertNewLine() {
	// Ensure cursor row is valid
	if f.cursor.Row < 0 {
		f.cursor.Row = 0
	}
	if f.cursor.Row >= len(f.lines) {
		for i := len(f.lines); i <= f.cursor.Row; i++ {
			f.lines = append(f.lines, "")
		}
	}

	line := f.lines[f.cursor.Row]

	// Clamp column
	if f.cursor.Col < 0 {
		f.cursor.Col = 0
	}
	if f.cursor.Col > len(line) {
		f.cursor.Col = len(line)
	}

	// Text after the cursor on the current line
	tail := line[f.cursor.Col:]

	// Text before the cursor becomes the current line content
	f.lines[f.cursor.Row] = line[:f.cursor.Col]

	// Insert the new line with the tail content efficiently
	// Create a new slice with capacity for one new element
	newLines := make([]string, len(f.lines)+1)
	copy(newLines, f.lines[:f.cursor.Row+1])
	newLines[f.cursor.Row+1] = tail
	copy(newLines[f.cursor.Row+2:], f.lines[f.cursor.Row+1:])
	f.lines = newLines

	// Move cursor to the beginning of the new line
	f.cursor.Row++
	f.cursor.Col = 0
	f.dirty = true
	f.adjustViewOffset() // Adjust viewport after adding a line
}

// DeleteRune deletes the character before the cursor (Backspace behavior)
func (f *Framebuffer) DeleteRune() {
	// Handle joining lines if at the start of a line (and not the first line)
	if f.cursor.Col == 0 && f.cursor.Row > 0 {
		prevLineIdx := f.cursor.Row - 1
		if prevLineIdx < 0 {
			return
		} // Should be caught by Row > 0, but safety

		prevLine := f.lines[prevLineIdx]
		currentLine := f.lines[f.cursor.Row]

		// New cursor position will be at the end of the joined previous line
		newCol := len(prevLine)

		// Join lines
		f.lines[prevLineIdx] = prevLine + currentLine

		// Remove current line efficiently
		copy(f.lines[f.cursor.Row:], f.lines[f.cursor.Row+1:])
		f.lines = f.lines[:len(f.lines)-1]

		// Move cursor to join point
		f.cursor.Row = prevLineIdx
		f.cursor.Col = newCol
		f.dirty = true
		f.adjustViewOffset() // Adjust viewport
		return
	}

	// Normal case: delete within the same line (if not at column 0)
	if f.cursor.Col > 0 && f.cursor.Row >= 0 && f.cursor.Row < len(f.lines) {
		line := f.lines[f.cursor.Row]
		// Ensure cursor column is not out of bounds (safety check)
		if f.cursor.Col > len(line) {
			f.cursor.Col = len(line) // Should not happen if MoveCursor clamps correctly
		}
		// If still at col 0 after clamping/checks, nothing to delete here
		if f.cursor.Col == 0 {
			return
		}

		newLine := line[:f.cursor.Col-1] + line[f.cursor.Col:]
		f.lines[f.cursor.Row] = newLine
		f.cursor.Col--
		f.dirty = true
		// View offset usually doesn't need adjusting for same-line delete
		// unless horizontal scrolling is implemented.
	}
	// If cursor.Col was 0 and cursor.Row was 0, do nothing.
}

// MoveCursor moves the cursor to a new position, adjusting bounds and viewport
func (f *Framebuffer) MoveCursor(row, col int) {
	// Ensure lines slice is never nil, always has at least one line
	if len(f.lines) == 0 {
		f.lines = []string{""}
	}

	// Clamp row
	row = min(row, len(f.lines)-1)
	row = max(row, 0)

	// Clamp column based on target row content
	// Target row must be valid after clamping
	lineLen := 0
	if row >= 0 && row < len(f.lines) {
		lineLen = len(f.lines[row])
	} // else lineLen remains 0

	col = min(col, lineLen)
	col = max(col, 0)

	// Only mark dirty and adjust view if position actually changes
	if f.cursor.Row != row || f.cursor.Col != col {
		f.cursor.Row = row
		f.cursor.Col = col
		f.dirty = true
		f.adjustViewOffset() // Adjust viewport if necessary
	}
}

// adjustViewOffset scrolls the viewport if the cursor moved out of view
func (f *Framebuffer) adjustViewOffset() {
	// Visible lines for content (terminal height minus footer line)
	visibleLines := f.height - 1
	if visibleLines <= 0 {
		visibleLines = 1 // Ensure at least one line is visible
	}

	needsAdjust := false
	newViewRow := f.viewOffset.Row

	// Scroll up if cursor is above the viewport
	if f.cursor.Row < f.viewOffset.Row {
		newViewRow = f.cursor.Row
		needsAdjust = true
	} else if f.cursor.Row >= f.viewOffset.Row+visibleLines {
		// Scroll down if cursor is below the viewport
		newViewRow = f.cursor.Row - visibleLines + 1
		needsAdjust = true
	}

	// TODO: Adjust horizontal offset (viewOffset.Col) if needed later
	// needsAdjust = needsAdjust || adjustHorizontalOffset()

	if needsAdjust {
		f.viewOffset.Row = newViewRow
		// Clamp viewOffset row just in case
		if f.viewOffset.Row < 0 {
			f.viewOffset.Row = 0
		}
		// Don't let viewOffset scroll too far down past the content
		maxViewRow := max(len(f.lines)-visibleLines, 0)
		if f.viewOffset.Row > maxViewRow {
			f.viewOffset.Row = maxViewRow
		}

		f.dirty = true // View offset change requires redraw
	}
}

// SetMode changes the editor mode
func (f *Framebuffer) SetMode(mode int) {
	if f.mode != mode {
		f.mode = mode
		f.dirty = true
		// Clear status message when mode changes (unless entering command mode?)
		// if mode != ModeCommand {
		//  f.statusMessage = ""
		// }
	}
}

// SetStatusMessage sets a message to be displayed temporarily in the footer.
func (f *Framebuffer) SetStatusMessage(msg string) {
	f.statusMessage = msg
	f.statusMessageTime = time.Now()
	f.dirty = true // Mark dirty to ensure footer redraws
}

// RenderTo generates ANSI commands to update the terminal display differentially,
// writing directly to the provided writer.
func (f *Framebuffer) RenderTo(writer *bufio.Writer) error {
	// Check if rendering is needed at all
	if !f.dirty && !f.forceRedraw {
		// Optimization: If nothing (content, cursor, mode, view, size) changed,
		// we might not even need to position the cursor. However, external
		// terminal interactions could move the cursor, so repositioning is safer.
		// If cursor position hasn't changed either, we could potentially return nil.
		if f.cursor == f.prevCursor { // Add this check for further optimization
			// return nil // Uncomment cautiously if cursor stability is guaranteed
		}
		// If only cursor moved, just update its position
		f.setCursorPosition(writer, f.cursor.Row, f.cursor.Col)
		f.prevCursor = f.cursor // Update previous cursor state
		return nil
	}

	// Use cached dimensions
	currentWidth := f.width
	currentHeight := f.height

	var err error

	// Check for resize or forced redraw
	if f.forceRedraw || currentWidth != f.prevWidth || currentHeight != f.prevHeight {
		err = f.renderFull(writer, currentWidth, currentHeight)
		f.forceRedraw = false // Reset flag after full redraw
	} else {
		// Perform differential rendering
		err = f.renderDiff(writer, currentWidth, currentHeight)
	}

	if err != nil {
		// Handle error from rendering helpers (e.g., write error)
		return err // Propagate error
	}

	// --- Update previous state for the next render cycle ---
	f.prevWidth = currentWidth
	f.prevHeight = currentHeight

	// Carefully copy the state of lines that were *visible* in this frame
	// Calculate visible lines based on *current* height
	visibleContentLines := max(currentHeight-1, 0)

	startRow := f.viewOffset.Row
	endRow := min(startRow+visibleContentLines, len(f.lines))

	numRenderedLines := 0
	if endRow > startRow {
		numRenderedLines = endRow - startRow
	}

	// Ensure prevLines slice has correct size
	if cap(f.prevLines) < numRenderedLines {
		f.prevLines = make([]string, numRenderedLines)
	} else {
		f.prevLines = f.prevLines[:numRenderedLines] // Reslice to the exact size needed
	}

	// Copy the content of the lines that were actually visible/rendered in *this* frame
	if numRenderedLines > 0 {
		copy(f.prevLines, f.lines[startRow:endRow])
	}

	// Store other previous states
	f.prevCursor = f.cursor
	f.prevMode = f.mode
	f.prevViewOffset = f.viewOffset
	f.prevStatusMessage = f.statusMessage
	f.prevStatusMessageTime = f.statusMessageTime
	// Save previous explorer state
	f.prevExplorerVisible = f.explorerVisible
	f.prevExplorerPath = f.explorerPath
	// Note: Shallow copy of entries might be okay if editor ensures slice replacement on change
	f.prevExplorerEntries = f.explorerEntries
	f.prevExplorerSelectedIndex = f.explorerSelectedIndex
	f.prevExplorerError = f.explorerError

	// Mark as clean *only* if no errors occurred and rendering completed
	f.dirty = false

	// The final writer.Flush() happens in Terminal.Run after RenderTo returns
	return nil
}

// renderFull performs a complete redraw of the screen
func (f *Framebuffer) renderFull(writer *bufio.Writer, width, height int) error {
	var err error

	_, err = writer.WriteString(hideCursor)
	if err != nil {
		return err
	}

	// Defer showing cursor until the function returns (successfully or not)
	// This ensures the cursor becomes visible even if errors occur mid-render.
	defer func() {
		// In explorer mode, we might want the cursor hidden or at a specific spot?
		// For now, always show it at the buffer position (which might be irrelevant in explorer)
		// Let's hide it completely in explorer mode for simplicity initially.
		if f.mode != ModeExplorer {
			_, showErr := writer.WriteString(showCursor)
			if err == nil && showErr != nil { // Capture the first error
				err = showErr
			}
		}
	}()

	// Clear screen and move cursor to home
	if _, err = writer.WriteString(clearScreenHome); err != nil {
		return err
	}

	// Render components based on mode
	if f.mode == ModeExplorer && f.explorerVisible {
		if err = f.renderExplorerView(writer, width, height); err != nil {
			return err
		}
	} else {
		// Render normal buffer content
		if err = f.renderAllContent(writer, width, height); err != nil {
			return err
		}
	}

	if err = f.renderFooter(writer, width, height); err != nil {
		return err
	}

	// After drawing everything, explicitly set the cursor position (if not explorer)
	if f.mode != ModeExplorer {
		if err = f.setCursorPosition(writer, f.cursor.Row, f.cursor.Col); err != nil {
			return err
		}
	}

	return err // Return the last error occurred
}

// renderDiff performs a differential update of the screen
func (f *Framebuffer) renderDiff(writer *bufio.Writer, width, height int) error {
	var err error
	// Hide cursor during diff updates to prevent flickering
	if _, err = writer.WriteString(hideCursor); err != nil {
		return err
	}
	defer func() {
		// Again, only show cursor if not in explorer mode
		if f.mode != ModeExplorer {
			_, showErr := writer.WriteString(showCursor)
			if err == nil && showErr != nil {
				err = showErr
			}
		}
	}()

	// Compare and update content based on mode
	if f.mode == ModeExplorer && f.explorerVisible {
		// TODO: Implement differential rendering for explorer view?
		// For now, just redraw the whole explorer view on any change.
		if err = f.renderExplorerView(writer, width, height); err != nil {
			return err
		}
	} else if f.prevMode == ModeExplorer && f.mode != ModeExplorer {
		// If we *just* switched *out* of explorer mode, force a full content redraw
		if err = f.renderAllContent(writer, width, height); err != nil {
			return err
		}
	} else if f.mode != ModeExplorer {
		// Perform differential rendering for normal buffer content
		if err = f.renderChangedContent(writer, width, height); err != nil {
			return err
		}
	}

	// Compare and update footer if needed (independent of main view)
	footerNeedsRedraw := f.mode != f.prevMode ||
		f.cursor != f.prevCursor || // Cursor pos might still be relevant for footer display
		f.viewOffset != f.prevViewOffset || // View offset also relevant
		(f.mode == ModeCommand) || // Always redraw footer in command mode
		(f.statusMessage != f.prevStatusMessage) || // Status message changed
		(f.statusMessage != "" && time.Since(f.statusMessageTime) >= statusMessageTimeout) || // Msg expired
		(f.prevStatusMessage != "" && time.Since(f.prevStatusMessageTime) < statusMessageTimeout) // Prev msg was active

	if footerNeedsRedraw {
		if err = f.renderFooter(writer, width, height); err != nil {
			return err
		}
	}

	// Crucially, update the physical cursor position if it changed AND not in explorer mode
	if f.mode != ModeExplorer && f.cursor != f.prevCursor {
		if err = f.setCursorPosition(writer, f.cursor.Row, f.cursor.Col); err != nil {
			return err
		}
	}

	return err // Return last error occurred
}

// renderExplorerView draws the file explorer content.
func (f *Framebuffer) renderExplorerView(writer *bufio.Writer, width, height int) error {
	visibleLines := height - 1 // Account for footer
	if visibleLines <= 0 {
		visibleLines = 1
	}

	var err error

	// Optional: Header for the explorer view
	header := "Explorer: " + f.explorerPath
	if f.explorerError != "" {
		header += " [ERROR: " + f.explorerError + "]"
	}

	// Simple header rendering (could be improved)
	if visibleLines > 0 {
		if _, err = writer.WriteString("\033[1;1H"); err != nil {
			return err
		} // Move to top-left
		if _, err = writer.WriteString(clearLine); err != nil {
			return err
		}
		// Truncate header if needed
		headerRunes := []rune(header)
		if len(headerRunes) > width {
			header = string(headerRunes[:width])
		}
		if _, err = writer.WriteString(reverseVideo); err != nil {
			return err
		}
		if _, err = writer.WriteString(header); err != nil {
			return err
		}
		if _, err = writer.WriteString(resetAttrs); err != nil {
			return err
		}
		visibleLines-- // Reduce available lines for entries
	}

	// TODO: Implement scrolling within the explorer list if it exceeds visibleLines
	startEntry := 0
	endEntry := min(len(f.explorerEntries), visibleLines)

	// Draw directory entries
	for i := startEntry; i < endEntry; i++ {
		screenRow := (i - startEntry) + 2 // 1-based screen row, +1 for 1-based, +1 for header

		// Move cursor to start of the screen line
		f.numBuf = f.numBuf[:0]
		f.numBuf = append(f.numBuf, '\033', '[')
		f.numBuf = strconv.AppendInt(f.numBuf, int64(screenRow), 10)
		f.numBuf = append(f.numBuf, ";1H"...)
		if _, err = writer.Write(f.numBuf); err != nil {
			return err
		}

		if _, err = writer.WriteString(clearLine); err != nil {
			return err
		}

		entry := f.explorerEntries[i]
		entryName := entry.Name()
		if entry.IsDir() {
			entryName += "/" // Indicate directory
		}

		// Apply highlighting for selected item
		isSelected := (i == f.explorerSelectedIndex)
		if isSelected {
			if _, err = writer.WriteString(reverseVideo); err != nil {
				return err
			}
		}

		// Truncate entry name if needed
		entryRunes := []rune(entryName)
		if len(entryRunes) > width {
			entryName = string(entryRunes[:width])
		}
		if _, err = writer.WriteString(entryName); err != nil {
			if isSelected {
				writer.WriteString(resetAttrs)
			} // Ensure reset on error
			return err
		}

		if isSelected {
			if _, err = writer.WriteString(resetAttrs); err != nil {
				return err
			}
		}
	}

	// Clear remaining lines below the entries (if any)
	startClearRow := endEntry + 2                                        // Screen row after the last entry
	for screenRow := startClearRow; screenRow <= height-1; screenRow++ { // height-1 because footer is last line
		// Move cursor
		f.numBuf = f.numBuf[:0]
		f.numBuf = append(f.numBuf, '\033', '[')
		f.numBuf = strconv.AppendInt(f.numBuf, int64(screenRow), 10)
		f.numBuf = append(f.numBuf, ";1H"...)
		if _, err = writer.Write(f.numBuf); err != nil {
			return err
		}
		// Clear line
		if _, err = writer.WriteString(clearLine); err != nil {
			return err
		}
	}

	return nil
}

// renderAllContent draws the entire visible text area for a full redraw
func (f *Framebuffer) renderAllContent(writer *bufio.Writer, width, height int) error {
	visibleLines := height - 1 // Account for footer only
	if visibleLines <= 0 {
		visibleLines = 1
	}

	startLine := f.viewOffset.Row
	endLine := min(startLine+visibleLines, len(f.lines))
	if startLine < 0 {
		startLine = 0
	} // Should not happen if viewOffset is clamped

	var err error
	// Draw lines within the visible range
	for i := startLine; i < endLine; i++ {
		screenRow := (i - startLine) + 1 // 1-based screen row

		// Move cursor to start of the screen line
		f.numBuf = f.numBuf[:0]
		f.numBuf = append(f.numBuf, '\033', '[')
		f.numBuf = strconv.AppendInt(f.numBuf, int64(screenRow), 10)
		f.numBuf = append(f.numBuf, ";1H"...) // Use WriteString directly for constant part? Saves append.
		_, err = writer.Write(f.numBuf)
		if err != nil {
			return err
		}

		// Clear the line before writing new content
		_, err = writer.WriteString(clearLine)
		if err != nil {
			return err
		}

		// Render the line content itself
		// TODO: Subtract width used by line numbers if they are added
		err = f.renderLineContent(writer, f.lines[i], width)
		if err != nil {
			return err
		}
	}

	// Fill remaining screen rows below content with tilde lines
	startTildeScreenRow := (endLine - startLine) + 1
	for screenRow := startTildeScreenRow; screenRow <= visibleLines; screenRow++ {
		// Move cursor to start of the screen line
		f.numBuf = f.numBuf[:0]
		f.numBuf = append(f.numBuf, '\033', '[')
		f.numBuf = strconv.AppendInt(f.numBuf, int64(screenRow), 10)
		f.numBuf = append(f.numBuf, ";1H"...)
		if _, err = writer.Write(f.numBuf); err != nil {
			return err
		}

		// Clear the line and write tilde
		if _, err = writer.WriteString(clearLine); err != nil {
			return err
		}

		if _, err = writer.WriteString(grayText); err != nil {
			return err
		}

		if _, err = writer.WriteString("~"); err != nil {
			return err
		}

		if _, err = writer.WriteString(resetAttrs); err != nil {
			return err
		}
	}
	return nil // No error found
}

// renderChangedContent draws only the changed lines for a differential update
func (f *Framebuffer) renderChangedContent(writer *bufio.Writer, width, height int) error {
	visibleLines := height - 1 // Current visible content lines
	if visibleLines <= 0 {
		visibleLines = 1
	}

	// Previous visible content lines (needs prevHeight)
	prevVisibleLines := f.prevHeight - 1
	if prevVisibleLines <= 0 {
		prevVisibleLines = 1
	}

	currentStart := f.viewOffset.Row

	numPrevLinesStored := len(f.prevLines)

	var err error

	// Iterate through the maximum possible screen rows that could have been affected
	maxScreenRowsToCheck := max(visibleLines, prevVisibleLines)

	for screenRowIdx := range maxScreenRowsToCheck {
		screenRow := screenRowIdx + 1 // 1-based screen row

		// --- Determine Current Line and if it's on the current screen ---
		currentBufferIdx := currentStart + screenRowIdx
		currentLine := ""
		currentLineExists := currentBufferIdx >= 0 && currentBufferIdx < len(f.lines)
		isOnCurrentScreen := screenRowIdx < visibleLines
		if currentLineExists && isOnCurrentScreen {
			currentLine = f.lines[currentBufferIdx]
		}

		// --- Determine Previous Line and if it was on the previous screen ---
		// Index within the prevLines slice corresponds to the screen row index *relative to the previous viewport start*
		prevStoredIdx := screenRowIdx // Index within the prevLines slice, IF it corresponds to the same screen line
		prevLine := ""
		prevLineExists := prevStoredIdx >= 0 && prevStoredIdx < numPrevLinesStored
		isOnPrevScreen := screenRowIdx < prevVisibleLines
		if prevLineExists && isOnPrevScreen {
			// We need to ensure the prevStoredIdx maps correctly to the buffer index based on currentStart
			// The way prevLines is stored (copying visible lines), prevStoredIdx *is* the correct index.
			prevLine = f.prevLines[prevStoredIdx]
		} else {
			// If it wasn't on the previous screen or the index is invalid,
			// consider it non-existent for comparison purposes.
			prevLineExists = false
		}

		// --- Decide if this screen line needs an update ---
		needsUpdate := false
		// Condition 1: Content mismatch on a line visible in *both* frames.
		if isOnCurrentScreen && isOnPrevScreen && currentLineExists && prevLineExists && currentLine != prevLine {
			needsUpdate = true
		} else if isOnCurrentScreen && !isOnPrevScreen {
			// Condition 2: Line is visible now but wasn't before (e.g., scroll down, resize larger).
			needsUpdate = true
		} else if !isOnCurrentScreen && isOnPrevScreen {
			// Condition 3: Line was visible before but isn't now (e.g., scroll up, resize smaller).
			// We need to clear it (or draw tilde if appropriate, but clear is safer).
			needsUpdate = true
		} else if isOnCurrentScreen && isOnPrevScreen && (currentLineExists != prevLineExists) {
			// Condition 4: Line position was visible in both frames, but content appeared/disappeared
			// (e.g., deleting lines below viewport caused tilde to appear/disappear).
			needsUpdate = true
		}

		// --- Perform Update if Needed ---
		if needsUpdate {
			// Move cursor to the start of the screen line to update
			f.numBuf = f.numBuf[:0]
			f.numBuf = append(f.numBuf, '\033', '[')
			f.numBuf = strconv.AppendInt(f.numBuf, int64(screenRow), 10)
			f.numBuf = append(f.numBuf, ";1H"...)
			if _, err = writer.Write(f.numBuf); err != nil {
				return err
			}

			// Clear the entire line first
			if _, err = writer.WriteString(clearLine); err != nil {
				return err
			}

			// Now, draw the correct content if this line *is* currently visible
			if isOnCurrentScreen {
				if currentLineExists {
					// Render the current line's content
					// TODO: Adjust width for line numbers if added
					if err = f.renderLineContent(writer, currentLine, width); err != nil {
						return err
					}
				} else {
					// This screen line is below the current content, draw tilde
					if _, err = writer.WriteString(grayText); err != nil {
						return err
					}
					if _, err = writer.WriteString("~"); err != nil {
						return err
					}
					if _, err = writer.WriteString(resetAttrs); err != nil {
						return err
					}
				}
			} // Else (isOnPrevScreen but !isOnCurrentScreen): Line is already cleared, nothing more to do.
		}
	}

	return nil // No error occurred
}

// renderLineContent draws the actual text content of a single line.
// Handles basic truncation for now.
func (f *Framebuffer) renderLineContent(writer *bufio.Writer, line string, maxWidth int) error {
	// TODO: Handle horizontal scrolling (viewOffset.Col)
	// TODO: Handle line wrapping more gracefully if desired.

	if maxWidth < 0 {
		maxWidth = 0
	} // Ensure maxWidth is not negative

	// Use runes for correct length calculation with unicode
	runes := []rune(line)
	lineLen := len(runes)

	// Simple truncation
	if lineLen <= maxWidth {
		_, err := writer.WriteString(line)
		return err
	} else {
		// Write only the truncated portion
		_, err := writer.WriteString(string(runes[:maxWidth]))
		return err
	}
}

// renderFooter draws the bottom status/command line directly to the writer.
func (f *Framebuffer) renderFooter(writer *bufio.Writer, width, height int) (err error) {
	// Move cursor to the beginning of the last line (height is 1-based)
	f.numBuf = f.numBuf[:0]
	f.numBuf = append(f.numBuf, '\033', '[')
	f.numBuf = strconv.AppendInt(f.numBuf, int64(height), 10)
	f.numBuf = append(f.numBuf, ";1H"...)
	if _, err := writer.Write(f.numBuf); err != nil {
		return err
	}

	// Clear the line
	if _, err = writer.WriteString(clearLine); err != nil {
		return err
	}

	// --- Determine Content ---
	var leftSide, rightSide string

	// Display status message if present and not too old
	if f.statusMessage != "" && time.Since(f.statusMessageTime) < statusMessageTimeout {
		leftSide = f.statusMessage
		// Clear the message after it's been displayed once and timeout passed?
		// Or maybe clear on next keypress instead?
		// For now, let it persist until timeout or replaced.
	} else if f.mode == ModeCommand {
		// Display command buffer content directly
		leftSide = ":" + f.commandBuffer
	} else {
		// Display mode indicator
		modeStr := ""
		switch f.mode {
		case ModeNormal:
			modeStr = "NORMAL"
		case ModeInsert:
			modeStr = "INSERT"
		case ModeVisual:
			modeStr = "VISUAL"
		default:
			modeStr = "UNKNOWN"
		}
		// Use bytes.Buffer for efficient string building here if complex
		leftSide = fmt.Sprintf("-- %s --", modeStr) // Sprintf okay for simple one-off?
	}

	// Right side: Cursor position, view offset, lines count
	// Use cached width/height (passed as args)
	// Build using strconv to avoid fmt.Sprintf overhead if performance critical
	// For now, Sprintf might be acceptable balance for readability here.
	rightSide = fmt.Sprintf("Ln %d, Col %d | View: %d | Size: %dx%d | Lines: %d",
		f.cursor.Row+1, f.cursor.Col+1, f.viewOffset.Row+1, // Show 1-based row/view
		width, height, len(f.lines))

	// --- Calculate Padding & Truncation ---
	leftLen := len([]rune(leftSide)) // Use rune length for accurate width
	rightLen := len([]rune(rightSide))

	// Available space for padding
	paddingNeeded := max(width-leftLen-rightLen, 1)

	// --- Write Content ---
	if _, err = writer.WriteString(reverseVideo); err != nil {
		return err
	}

	// Write left side (potentially truncated if very long)
	written := 0
	for _, r := range leftSide { // Iterate directly over the string
		if written >= width {
			break
		}
		if _, err = writer.WriteRune(r); err != nil {
			return err
		}
		written++
	}

	// Write padding
	if written < width {
		spacesToWrite := paddingNeeded
		// Adjust padding if left+right already exceeds width
		if leftLen+rightLen >= width {
			spacesToWrite = 1 // Write just one space if total exceeds width
		}
		if written+spacesToWrite > width {
			spacesToWrite = width - written // Don't write more padding than available space
		}
		for range spacesToWrite {
			if err = writer.WriteByte(' '); err != nil {
				return err
			}
			written++
		}
	}

	// Write right side (truncated to fit remaining space)
	if written < width {
		for _, r := range rightSide { // Iterate directly over the string
			if written >= width {
				break
			}
			if _, err = writer.WriteRune(r); err != nil {
				return err
			}
			written++
		}
	}

	// Reset attributes
	if _, err = writer.WriteString(resetAttrs); err != nil {
		return err
	}
	return err
}

// IsDirty checks if the framebuffer requires rendering.
// Now includes checks for size changes implicitly covered by forceRedraw.
func (f *Framebuffer) IsDirty() bool {
	// If forced, it's dirty.
	// If content/cursor/mode/view changed since last render, it's dirty.
	// Check if a status message needs to be drawn or cleared.
	messageActive := f.statusMessage != ""
	messageExpired := messageActive && time.Since(f.statusMessageTime) >= statusMessageTimeout
	prevMessageWasActive := f.prevStatusMessage != ""
	prevMessageExpired := prevMessageWasActive && time.Since(f.prevStatusMessageTime) >= statusMessageTimeout

	return f.forceRedraw || f.dirty ||
		f.cursor != f.prevCursor ||
		f.mode != f.prevMode ||
		f.viewOffset != f.prevViewOffset ||
		f.width != f.prevWidth || // Check size change directly too
		f.height != f.prevHeight ||
		(messageActive != prevMessageWasActive) || // Message appeared or disappeared
		(messageActive && !messageExpired && f.statusMessage != f.prevStatusMessage) || // Message text changed while active
		(prevMessageWasActive && !prevMessageExpired && messageExpired) || // Message just expired
		// Explorer specific checks
		(f.explorerVisible != f.prevExplorerVisible) || // Explorer appeared/disappeared
		(f.explorerVisible && (f.explorerPath != f.prevExplorerPath || f.explorerSelectedIndex != f.prevExplorerSelectedIndex || f.explorerError != f.prevExplorerError))
}

// GetContent returns the buffer content as a string (primarily for saving/io.Reader)
func (f *Framebuffer) GetContent() string {
	// TODO: Consider using a strings.Builder for efficiency if many lines
	return strings.Join(f.lines, "\n")
}

// SetContent sets the buffer content from a string, replacing current content.
func (f *Framebuffer) SetContent(content string) {
	// Split content into lines
	f.lines = strings.Split(content, "\n")
	// Ensure there's always at least one line, even if content is empty
	if len(f.lines) == 0 {
		f.lines = []string{""}
	}
	// Reset state after loading content
	f.cursor = Position{0, 0}
	f.viewOffset = Position{0, 0}
	f.commandBuffer = ""
	f.dirty = true
	f.forceRedraw = true  // Force full redraw after setting content
	f.ClearExplorerData() // Ensure explorer is not visible when content is set externally
}

// SetExplorerData updates the framebuffer with data needed to render the explorer view.
func (f *Framebuffer) SetExplorerData(path string, entries []os.DirEntry, selectedIndex int, errorMsg string) {
	f.explorerVisible = true
	f.explorerPath = path
	f.explorerEntries = entries
	f.explorerSelectedIndex = selectedIndex
	f.explorerError = errorMsg
	f.dirty = true // Mark dirty to trigger redraw
}

// ClearExplorerData hides the explorer view and clears its data.
func (f *Framebuffer) ClearExplorerData() {
	f.explorerVisible = false
	f.explorerPath = ""
	f.explorerEntries = nil
	f.explorerSelectedIndex = -1 // Indicate no selection
	f.explorerError = ""
	f.dirty = true // Mark dirty to trigger redraw
}

// setCursorPosition generates the command to move the physical terminal cursor,
// writing it directly to the writer.
func (f *Framebuffer) setCursorPosition(writer *bufio.Writer, bufferRow, bufferCol int) error {
	// Calculate screen position based on view offset.
	// Screen rows/cols are 1-based.
	screenRow := (bufferRow - f.viewOffset.Row) + 1
	screenCol := bufferCol + 1 // TODO: Adjust for horizontal scroll (f.viewOffset.Col) later

	// Clamp cursor position to be within the visible text area bounds
	// Visible area is height - 1 (excluding footer)
	visibleContentLines := f.height - 1
	if visibleContentLines <= 0 {
		visibleContentLines = 1
	}

	minScreenRow := 1
	maxScreenRow := minScreenRow + visibleContentLines - 1
	minScreenCol := 1
	maxScreenCol := f.width // Max column is terminal width

	// Clamp screenRow
	if screenRow < minScreenRow {
		screenRow = minScreenRow
	}
	if screenRow > maxScreenRow {
		screenRow = maxScreenRow
	}

	// Clamp screenCol
	if screenCol < minScreenCol {
		screenCol = minScreenCol
	}
	if screenCol > maxScreenCol {
		screenCol = maxScreenCol
	}

	// Build ANSI escape sequence efficiently: \033[<row>;<col>H
	f.numBuf = f.numBuf[:0] // Clear shared buffer
	f.numBuf = append(f.numBuf, '\033', '[')
	f.numBuf = strconv.AppendInt(f.numBuf, int64(screenRow), 10)
	f.numBuf = append(f.numBuf, ';')
	f.numBuf = strconv.AppendInt(f.numBuf, int64(screenCol), 10)
	f.numBuf = append(f.numBuf, 'H')

	// Write the sequence to the buffered writer
	_, err := writer.Write(f.numBuf)
	return err
}
