package errnie

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/phuslu/log"
	"github.com/theapemachine/caramba/pkg/config"
)

var logger *Logger

var elasticsearchWriterWarn sync.Once

func init() {
	log.DefaultLogger = log.Logger{
		Level:      log.InfoLevel,
		Caller:     1,
		TimeField:  "date",
		TimeFormat: "2006-01-02 15:04:05",
		Writer:     log.IOWriter{Writer: os.Stdout},
	}

	logger = NewLogger()
}

/*
Apply reconfigures the global errnie logger from ErrnieConfig (call after viper is loaded).
*/
func Apply(cfg *config.ErrnieConfig) {
	log.DefaultLogger = log.Logger{
		Level:      parseLogLevel(cfg.Level),
		Caller:     1,
		TimeField:  "date",
		TimeFormat: "2006-01-02 15:04:05",
		Writer:     buildWriter(cfg),
	}
}

func buildWriter(cfg *config.ErrnieConfig) log.Writer {
	writers := make([]log.Writer, 0, 3)
	writers = append(writers, log.IOWriter{Writer: os.Stdout})

	if cfg.File.Active && strings.TrimSpace(cfg.File.Path) != "" {
		writers = append(writers, &log.FileWriter{
			Filename:     cfg.File.Path,
			EnsureFolder: true,
		})
	}

	if cfg.Elasticsearch.Active {
		elasticSink, err := newElasticPostWriter(
			cfg.Elasticsearch.URL,
			cfg.Elasticsearch.Index,
			cfg.Elasticsearch.Username,
			cfg.Elasticsearch.Password,
		)

		if err != nil {
			elasticsearchWriterWarn.Do(func() {
				fmt.Fprintf(os.Stderr, "errnie: %v\n", err)
			})
		} else {
			writers = append(writers, &log.AsyncWriter{
				Writer:        log.IOWriter{Writer: elasticSink},
				ChannelSize:   256,
				DiscardOnFull: true,
			})
		}
	}

	if len(writers) == 1 {
		return writers[0]
	}

	multi := log.MultiEntryWriter(writers)

	return &multi
}

func parseLogLevel(level string) log.Level {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "trace":
		return log.TraceLevel
	case "debug":
		return log.DebugLevel
	case "info", "":
		return log.InfoLevel
	case "warn", "warning":
		return log.WarnLevel
	case "error":
		return log.ErrorLevel
	case "fatal":
		return log.FatalLevel
	case "panic":
		return log.PanicLevel
	default:
		return log.InfoLevel
	}
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

The error is returned unmodified, which allows it to wrap and log the error
directly, preventing yet more repetitive error handling code.

Example:

```
func DoSomething() error {
	return fmt.Errorf("something went wrong: %w", errors.New("something else went wrong"))
}

func DoAndReturnSomething() (string, error) {
	return "something", errnie.Error(DoSomething())
}

func main() {
	errnie.Error(DoSomething())

	something, err := DoAndReturnSomething()

	if err != nil {
		fmt.Println("something went wrong:", err)
	}

	fmt.Println("something:", something)
}
```
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
