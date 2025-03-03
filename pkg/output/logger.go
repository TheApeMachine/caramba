package output

import (
	"os"
	"strings"
	"sync"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/theapemachine/caramba/pkg/hub"
)

var once sync.Once
var logger *Logger

type Logger struct {
	hub  *hub.Queue
	file *os.File
}

func NewLogger() *Logger {
	once.Do(func() {
		// First empty the log file
		os.Remove("debug.log")

		// Then create a new log file
		f, err := tea.LogToFile("debug.log", "debug")
		if err != nil {
			panic(err)
		}

		logger = &Logger{hub: hub.NewQueue(), file: f}
	})

	return logger
}

func (l *Logger) Log(origin string, message string) {
	l.file.WriteString(message + "\n")
	l.hub.Add(hub.NewStatus(origin, "unknown", strings.Join([]string{origin, message}, " ")))
}

func (l *Logger) Error(origin string, err error) error {
	if err == nil {
		return nil
	}

	l.file.WriteString(err.Error() + "\n")
	l.hub.Add(hub.NewStatus(origin, "error", strings.Join([]string{origin, err.Error()}, " ")))
	return err
}

func (l *Logger) Success(origin string, message string) {
	l.file.WriteString(message + "\n")
	l.hub.Add(hub.NewStatus(origin, "success", strings.Join([]string{origin, message}, " ")))
}

func (l *Logger) Close() {
	l.file.Close()
}
