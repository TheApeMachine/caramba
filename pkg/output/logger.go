package output

import (
	"os"
	"sync"

	tea "github.com/charmbracelet/bubbletea"
)

var once sync.Once
var logger *Logger

type Logger struct {
	file *os.File
}

func NewLogger() *Logger {
	once.Do(func() {
		os.Remove("debug.log")

		f, err := tea.LogToFile("debug.log", "debug")

		if err != nil {
			panic(err)
		}

		logger = &Logger{file: f}
	})

	return logger
}

func (l *Logger) Log(origin string, message string) {
	l.file.WriteString(message + "\n")
}

func (l *Logger) Error(origin string, err error) error {
	if err == nil {
		return nil
	}

	l.file.WriteString(err.Error() + "\n")
	return err
}

func (l *Logger) Warning(origin string, message string) {
	l.file.WriteString(message + "\n")
}

func (l *Logger) Success(origin string, message string) {
	l.file.WriteString(message + "\n")
}

func (l *Logger) Close() {
	l.file.Close()
}
