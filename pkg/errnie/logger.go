package errnie

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/log"
)

var logger = log.NewWithOptions(os.Stderr, log.Options{
	ReportCaller:    true,
	CallerOffset:    1,
	ReportTimestamp: true,
	TimeFormat:      time.TimeOnly,
})

func SetLevel(level log.Level) {
	logger.SetLevel(level)
}

func Error(msg any, keyvals ...any) (err error) {
	if msg == nil {
		return nil
	}

	// Log the error message and stack trace
	logger.Error(msg, keyvals...)
	fmt.Println(getStackTrace())

	// Return the original error
	if err, ok := msg.(error); ok {
		return err
	}

	for _, keyval := range keyvals {
		if err, ok := keyval.(error); ok {
			return err
		}
	}

	return
}

func Warn(msg any, keyvals ...any) {
	logger.Warn(msg, keyvals...)
}

func Info(msg any, keyvals ...any) {
	logger.Info(msg, keyvals...)
}

func Debug(msg any, keyvals ...any) {
	logger.Debug(msg, keyvals...)
}

/*
Retrieve and format a stack trace from the current execution point.
*/
func getStackTrace() string {
	const depth = 10
	var pcs [depth]uintptr
	n := runtime.Callers(3, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])

	var trace strings.Builder
	trace.WriteString("\n📚 Stack trace:\n")

	isFirst := true
	for {
		frame, more := frames.Next()
		if !more {
			break
		}

		// Skip standard library frames
		if strings.Contains(frame.File, "runtime/") || strings.Contains(frame.File, "/src/") {
			continue
		}

		// Just show the last part of the path for clarity
		file := filepath.Base(frame.File)
		funcName := filepath.Base(frame.Function)

		if frame.Line == 0 {
			continue
		}

		prefix := "   "
		if isFirst {
			prefix = "➜ " // Arrow pointing to the error origin
			isFirst = false
		}

		line := fmt.Sprintf("%s%s:%d %s\n",
			prefix,
			file,
			frame.Line,
			funcName,
		)
		trace.WriteString(line)

		// Show code snippet only for the first (error origin) frame
		if prefix == "➜ " {
			snippet := getCodeSnippet(frame.File, frame.Line, 2)
			trace.WriteString(snippet)
		}
	}

	return trace.String()
}

/*
Retrieve and return a code snippet surrounding the given line in the provided file.
*/
func getCodeSnippet(file string, errorLine, radius int) string {
	if file == "" {
		return ""
	}

	fileHandle, err := os.Open(file)
	if err != nil {
		return ""
	}
	defer fileHandle.Close()

	var snippet strings.Builder
	snippet.WriteString("\n📝 Code:\n")

	scanner := bufio.NewScanner(fileHandle)
	currentLine := 1

	for scanner.Scan() && currentLine <= errorLine+radius {
		if currentLine >= errorLine-radius {
			lineText := scanner.Text()
			prefix := "   "
			if currentLine == errorLine {
				prefix = "➜ " // Arrow pointing to the error line
				lineText = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF5555")).Render(lineText)
			}
			snippet.WriteString(fmt.Sprintf("%s%s\n", prefix, lineText))
		}
		currentLine++
	}

	return snippet.String()
}
