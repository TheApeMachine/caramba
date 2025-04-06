package errnie

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/log"
)

var (
	output      = true
	writeToFile = true
	mu          sync.RWMutex
	logFile     = "caramba.log"
	fileHandle  *os.File
)

var logger = log.NewWithOptions(os.Stderr, log.Options{
	ReportCaller:    true,
	CallerOffset:    1,
	ReportTimestamp: true,
	TimeFormat:      time.TimeOnly,
})

func init() {
	var err error

	if fileHandle, err = os.OpenFile(
		logFile, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644,
	); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open log file: %v\n", err)
		return
	}
}

// Cleanup closes the log file handle
func Cleanup() {
	mu.Lock()
	defer mu.Unlock()

	if fileHandle != nil {
		fileHandle.Close()
		fileHandle = nil
	}
}

// SetLevel sets the logging level
func SetLevel(level log.Level) {
	logger.SetLevel(level)
}

func SetOutput(out bool) {
	mu.Lock()
	defer mu.Unlock()

	output = out
}

func Log(msg any, keyvals ...any) {
	if writeToFile {
		WriteToFile(msg, keyvals...)
	}
}

// Error logs an error message and returns the original error
func Error(msg any, keyvals ...any) (err error) {
	if msg == nil {
		return nil
	}

	if writeToFile {
		WriteToFile(msg, keyvals...)
	}

	if !output {
		return
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

// Warn logs a warning message
func Warn(msg any, keyvals ...any) {
	if writeToFile {
		WriteToFile(msg, keyvals...)
	}

	if !output {
		return
	}

	logger.Warn(msg, keyvals...)
}

// Info logs an informational message
func Info(msg any, keyvals ...any) {
	if writeToFile {
		WriteToFile(msg, keyvals...)
	}

	if !output {
		return
	}

	logger.Info(msg, keyvals...)
}

// Debug logs a debug message
func Debug(msg any, keyvals ...any) {
	if writeToFile {
		WriteToFile(msg, keyvals...)
	}

	if !output {
		return
	}

	logger.Debug(msg, keyvals...)
}

// WriteToFile writes log messages to a file
func WriteToFile(msg any, keyvals ...any) {
	mu.Lock()
	defer mu.Unlock()

	if fileHandle == nil {
		var err error
		fileHandle, err = os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return
		}
	}

	// Get caller info
	pc, file, line, ok := runtime.Caller(2) // Skip 2 frames to get actual caller
	if !ok {
		file = "unknown"
		line = 0
	}

	funcName := runtime.FuncForPC(pc).Name()
	if idx := strings.LastIndex(funcName, "/"); idx >= 0 {
		funcName = funcName[idx+1:]
	}

	// Format key-value pairs
	var kvStr string
	if len(keyvals) > 0 {
		pairs := make([]string, 0, len(keyvals)/2)
		for i := 0; i < len(keyvals); i += 2 {
			key := fmt.Sprintf("%v", keyvals[i])
			var val string
			if i+1 < len(keyvals) {
				val = fmt.Sprintf("%v", keyvals[i+1])
			}
			pairs = append(pairs, fmt.Sprintf("%s=%s", key, val))
		}
		kvStr = " " + strings.Join(pairs, " ")
	}

	// Format log entry to match console output
	logEntry := fmt.Sprintf("%s %s %s:%d %s%s\n",
		time.Now().Format(time.RFC3339),
		"["+strings.ReplaceAll(file, "/Users/theapemachine/go/src/github.com/theapemachine/caramba/", "")+":"+strconv.Itoa(line)+"]",
		"<"+funcName+">",
		line,
		msg,
		kvStr,
	)

	if _, err := fileHandle.WriteString(logEntry); err != nil {
		// If write failed, try to reopen the file
		fileHandle.Close()
		fileHandle = nil
	}
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
