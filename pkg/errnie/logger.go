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
	TimeFormat:      time.RFC3339,
})

func SetLevel(level log.Level) {
	logger.SetLevel(level)
}

func Error(msg any, keyvals ...any) (err error) {
	if msg == nil {
		return nil
	}

	logger.Error(msg, keyvals...)
	//fmt.Println(getStackTrace())

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
	const depth = 32
	var pcs [depth]uintptr
	n := runtime.Callers(3, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])

	var trace strings.Builder
	for {
		frame, more := frames.Next()
		if !more {
			break
		}

		funcName := frame.Function

		if lastSlash := strings.LastIndexByte(funcName, '/'); lastSlash >= 0 {
			funcName = funcName[lastSlash+1:]
		}

		funcName = strings.Replace(funcName, ".", ":", 1)

		line := fmt.Sprintf("%s at %s(line %d)\n",
			lipgloss.NewStyle().Foreground(lipgloss.Color("#6E95F7")).Render(funcName),
			lipgloss.NewStyle().Foreground(lipgloss.Color("#06C26F")).Render(filepath.Base(frame.File)),
			frame.Line,
		)

		trace.WriteString(line)
		codeSnippet := getCodeSnippet(frame.File, frame.Line, 3)
		trace.WriteString(codeSnippet)
	}

	return "\n===[STACK TRACE]===\n" + trace.String() + "===[/STACK TRACE]===\n"
}

/*
Retrieve and return a code snippet surrounding the given line in the provided file.
*/
func getCodeSnippet(file string, line, radius int) string {
	if file == "" {
		return ""
	}

	fileHandle, err := os.Open(file)
	if err != nil {
		logger.Warn("Failed to open file for code snippet", "file", file, "error", err)
		return ""
	}
	defer fileHandle.Close()

	scanner := bufio.NewScanner(fileHandle)
	currentLine := 1
	var snippet strings.Builder

	for scanner.Scan() {
		if currentLine >= line-radius && currentLine <= line+radius {
			prefix := "  "
			if currentLine == line {
				prefix = "> "
			}
			snippet.WriteString(fmt.Sprintf("%s%d: %s\n", prefix, currentLine, scanner.Text()))
		}
		currentLine++
	}

	if err := scanner.Err(); err != nil {
		logger.Warn("Failed to read from code snippet file", "file", file, "error", err)
		return ""
	}

	return snippet.String()
}
