package tui

import (
	"regexp"
)

// ANSI escape sequence regex patterns
var ansiRegex = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
var comprehensiveAnsiRegex = regexp.MustCompile(`(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]`)

// StripANSI removes ANSI escape sequences from the provided string
func StripANSI(s string) string {
	return ansiRegex.ReplaceAllString(s, "")
}

// StripANSIComprehensive removes a wider range of ANSI escape sequences
// from the provided string, including more complex control sequences
func StripANSIComprehensive(s string) string {
	return comprehensiveAnsiRegex.ReplaceAllString(s, "")
}
