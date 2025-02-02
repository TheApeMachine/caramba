package utils

import (
	"strings"

	"github.com/acarl005/stripansi"
)

// CleanTerminalOutput strips ANSI sequences and problematic control characters
func CleanTerminalOutput(output string) string {
	// Strip ANSI sequences
	output = stripansi.Strip(output)

	// Remove bell character and other control chars that cause issues
	output = strings.ReplaceAll(output, "\u0007", "")  // Remove bell character
	output = strings.ReplaceAll(output, "\u003e", ">") // Fix escaped >
	output = strings.ReplaceAll(output, "\u003c", "<") // Fix escaped <
	output = strings.ReplaceAll(output, "\u0026", "&") // Fix escaped &

	// Clean up any double newlines or spaces
	output = strings.ReplaceAll(output, "\n\n\n", "\n\n")
	output = strings.ReplaceAll(output, "  ", " ")

	return strings.TrimSpace(output)
}
