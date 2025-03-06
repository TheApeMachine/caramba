package errnie

import (
	"os"
	"time"

	"github.com/charmbracelet/log"
)

var logger = log.NewWithOptions(os.Stderr, log.Options{
	ReportCaller:    true,
	CallerOffset:    1,
	ReportTimestamp: true,
	TimeFormat:      time.Kitchen,
})

func SetLevel(level log.Level) {
	logger.SetLevel(level)
}

func Error(msg any, keyvals ...any) (err error) {
	if msg == nil {
		return nil
	}

	var ok bool

	for _, keyval := range keyvals {
		if err, ok = keyval.(error); ok {
			return
		}
	}

	if err == nil {
		return
	}

	logger.Error(msg, keyvals...)
	return
}

func Debug(msg any, keyvals ...any) {
	logger.Debug(msg, keyvals...)
}
