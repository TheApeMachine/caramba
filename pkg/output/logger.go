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
		f, err := tea.LogToFile("debug.log", "debug")
		if err != nil {
			panic(err)
		}

		logger = &Logger{file: f}
	})

	return logger
}

func (l *Logger) Log(message string) {
	l.file.WriteString(message + "\n")
}

func (l *Logger) Close() {
	l.file.Close()
}
