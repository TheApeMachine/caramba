package output

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/briandowns/spinner"
	"github.com/fatih/color"
)

// OutputLevel controls how verbose the terminal output should be
type OutputLevel int

const (
	// OutputMinimal shows only essential information
	OutputMinimal OutputLevel = iota
	// OutputNormal shows moderate information (default)
	OutputNormal
	// OutputVerbose shows detailed information
	OutputVerbose
	// OutputDebug shows all available information
	OutputDebug
)

// Terminal provides methods for consistent terminal output formatting
type Terminal struct {
	// Current output level
	Level OutputLevel
	// Whether to use colors
	Colorized bool
	// Whether to show emojis
	Emojis bool
	// Active spinners
	spinners []*spinner.Spinner
}

// Global terminal instance with default settings
var term *Terminal

// Initialize the terminal on package import
func init() {
	// Default to normal output level
	level := OutputNormal

	// Check if CARAMBA_OUTPUT environment variable is set
	if os.Getenv("CARAMBA_OUTPUT") != "" {
		if l, err := strconv.Atoi(os.Getenv("CARAMBA_OUTPUT")); err == nil {
			level = OutputLevel(l)
		}
	}

	// Create the terminal with default settings
	term = &Terminal{
		Level:     level,
		Colorized: true,
		Emojis:    true,
	}
}

// SetLevel changes the output verbosity level
func SetLevel(level OutputLevel) {
	term.Level = level
}

// DisableColor turns off colored output
func DisableColor() {
	term.Colorized = false
	color.NoColor = true
}

// DisableEmojis turns off emoji output
func DisableEmojis() {
	term.Emojis = false
}

// formatWithEmoji prepends an emoji to text if enabled
func (t *Terminal) formatWithEmoji(emoji, text string) string {
	if t.Emojis {
		return emoji + " " + text
	}
	return text
}

// Title prints a large title with padding
func Title(text string) {
	if term.Level < OutputMinimal {
		return
	}

	width := 60
	fmt.Println()
	if term.Colorized {
		color.New(color.FgHiCyan, color.Bold).Println(strings.Repeat("═", width))
		color.New(color.FgHiCyan, color.Bold).Println(centerText(text, width))
		color.New(color.FgHiCyan, color.Bold).Println(strings.Repeat("═", width))
	} else {
		fmt.Println(strings.Repeat("═", width))
		fmt.Println(centerText(text, width))
		fmt.Println(strings.Repeat("═", width))
	}
	fmt.Println()
}

// centerText centers text within a given width
func centerText(text string, width int) string {
	if len(text) >= width {
		return text
	}

	padding := (width - len(text)) / 2
	return strings.Repeat(" ", padding) + text + strings.Repeat(" ", padding)
}

// Stage prints a stage header in the workflow
func Stage(number int, description string) {
	if term.Level < OutputNormal {
		return
	}

	emoji := "🔄"
	switch number {
	case 1:
		emoji = "📋"
	case 2:
		emoji = "🌐"
	case 3:
		emoji = "🧠"
	case 4:
		emoji = "📊"
	}

	text := term.formatWithEmoji(emoji, fmt.Sprintf("STAGE %d: %s", number, description))

	if term.Colorized {
		color.New(color.FgGreen, color.Bold).Println(text)
	} else {
		fmt.Println(text)
	}
}

// Action prints an action being taken
func Action(tool, action string, detail string) {
	if term.Level < OutputNormal {
		return
	}

	var emoji string
	switch tool {
	case "browser":
		emoji = "🌐"
	case "memory":
		emoji = "📝"
	case "agent":
		emoji = "🧠"
	case "calculator":
		emoji = "🧮"
	default:
		emoji = "🛠️"
	}

	text := term.formatWithEmoji(emoji, fmt.Sprintf("%s: %s %s", tool, action, detail))

	if term.Colorized {
		color.New(color.FgBlue).Println(text)
	} else {
		fmt.Println(text)
	}
}

// Result prints a result from an operation
func Result(message string) {
	if term.Level < OutputNormal {
		return
	}

	text := term.formatWithEmoji("✅", message)

	if term.Colorized {
		color.New(color.FgGreen).Println(text)
	} else {
		fmt.Println(text)
	}
}

// Error prints an error message
func Error(message string, err error) {
	if term.Level < OutputMinimal {
		return
	}

	text := term.formatWithEmoji("❌", fmt.Sprintf("%s: %v", message, err))

	if term.Colorized {
		color.New(color.FgRed, color.Bold).Println(text)
	} else {
		fmt.Println(text)
	}
}

// Info prints an informational message
func Info(message string) {
	if term.Level < OutputNormal {
		return
	}

	text := term.formatWithEmoji("ℹ️", message)

	if term.Colorized {
		color.New(color.FgCyan).Println(text)
	} else {
		fmt.Println(text)
	}
}

// Warn prints a warning message
func Warn(message string) {
	if term.Level < OutputNormal {
		return
	}

	text := term.formatWithEmoji("⚠️", message)

	if term.Colorized {
		color.New(color.FgYellow, color.Bold).Println(text)
	} else {
		fmt.Println(text)
	}
}

// Debug prints a debug message only in debug mode
func Debug(message string) {
	if term.Level < OutputDebug {
		return
	}

	text := term.formatWithEmoji("🔍", message)

	if term.Colorized {
		color.New(color.FgHiBlack).Println(text)
	} else {
		fmt.Println(text)
	}
}

// StartSpinner creates and starts a spinner with the given message
func StartSpinner(message string) *spinner.Spinner {
	if term.Level < OutputNormal {
		return nil
	}

	s := spinner.New(spinner.CharSets[9], 100*time.Millisecond)

	if term.Emojis {
		s.Prefix = "🔄 "
	}

	s.Suffix = " " + message

	if term.Colorized {
		s.Color("blue")
	}

	s.Start()
	term.spinners = append(term.spinners, s)
	return s
}

// StopSpinner stops a spinner and prints a completion message
func StopSpinner(s *spinner.Spinner, message string) {
	if s == nil {
		return
	}

	s.Stop()
	if message != "" {
		Result(message)
	}

	// Remove from active spinners
	for i, spin := range term.spinners {
		if spin == s {
			term.spinners = append(term.spinners[:i], term.spinners[i+1:]...)
			break
		}
	}
}

// FormatResearchResults formats a research result string with nicer formatting
func FormatResearchResults(results string) string {
	if !term.Colorized {
		return results
	}

	var sb strings.Builder

	lines := strings.Split(results, "\n")
	for i, line := range lines {
		// Format headings
		if strings.HasPrefix(line, "# ") {
			// Main heading
			sb.WriteString(color.New(color.FgHiCyan, color.Bold).Sprint(line) + "\n")
		} else if strings.HasPrefix(line, "## ") {
			// Sub heading
			sb.WriteString(color.New(color.FgCyan, color.Bold).Sprint(line) + "\n")
		} else if strings.HasPrefix(line, "### ") {
			// Sub-sub heading
			sb.WriteString(color.New(color.FgBlue, color.Bold).Sprint(line) + "\n")
		} else if i > 0 && strings.HasPrefix(lines[i-1], "#") {
			// Line after heading
			sb.WriteString(color.New(color.FgYellow).Sprint(line) + "\n")
		} else if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") {
			// List items
			sb.WriteString(color.New(color.FgGreen).Sprint(line) + "\n")
		} else if strings.Contains(line, "`") {
			// Lines containing code
			parts := strings.Split(line, "`")
			for j, part := range parts {
				if j%2 == 1 { // Inside backticks
					sb.WriteString(color.New(color.FgHiMagenta).Sprint("`" + part + "`"))
				} else { // Outside backticks
					sb.WriteString(part)
				}
			}
			sb.WriteString("\n")
		} else {
			// Normal text
			sb.WriteString(line + "\n")
		}
	}

	return sb.String()
}

// PrintResearchResults prints the full research results with formatting
func PrintResearchResults(results string) {
	if term.Level < OutputMinimal {
		return
	}

	Title("RESEARCH RESULTS")
	fmt.Println(FormatResearchResults(results))
	fmt.Println(strings.Repeat("═", 60))
}

// Verbose prints a message only in verbose or debug mode
func Verbose(message string) {
	if term.Level < OutputVerbose {
		return
	}

	text := term.formatWithEmoji("🔊", message)

	if term.Colorized {
		color.New(color.FgHiBlack).Println(text)
	} else {
		fmt.Println(text)
	}
}

// Summarize truncates a long string to a max length with ellipsis
func Summarize(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}

	// Try to cut at a space
	cutIdx := maxLen
	for i := maxLen; i > maxLen-10 && i > 0; i-- {
		if text[i] == ' ' {
			cutIdx = i
			break
		}
	}

	return text[:cutIdx] + "..."
}
