package errnie

import (
	"os"

	"github.com/phuslu/log"
)

var logger *Logger

func init() {
	log.DefaultLogger = log.Logger{
		Level:      log.InfoLevel,
		Caller:     1,
		TimeField:  "date",
		TimeFormat: "2006-01-02 15:04:05",
		Writer:     &log.IOWriter{Writer: os.Stdout},
	}

	logger = NewLogger()
}

/*
Logger is the main logger for the errnie package.
*/
type Logger struct {
	handle *log.Logger
}

/*
NewLogger creates a new Logger with the default logger.
*/
func NewLogger() *Logger {
	return &Logger{handle: &log.DefaultLogger}
}

/*
LogErr logs err at error level with optional alternating key/value fields.
*/
func Error(err error, fields ...any) error {
	if err != nil {
		logger.handle.Error().Err(err).KeysAndValues(fields).Msg(err.Error())
	}

	return err
}

/*
Warn logs message at warn level with optional alternating key/value fields.
*/
func Warn(message string, fields ...any) {
	logger.handle.Warn().KeysAndValues(fields).Msg(message)
}

/*
Info logs message at info level with optional alternating key/value fields.
*/
func Info(message string, fields ...any) {
	logger.handle.Info().KeysAndValues(fields).Msg(message)
}

/*
Debug logs message at debug level with optional alternating key/value fields.
*/
func Debug(message string, fields ...any) {
	logger.handle.Debug().KeysAndValues(fields).Msg(message)
}

/*
Trace logs message at trace level with optional alternating key/value fields.
*/
func Trace(message string, fields ...any) {
	logger.handle.Trace().KeysAndValues(fields).Msg(message)
}
