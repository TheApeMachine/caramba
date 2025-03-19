package cmd

import (
	"fmt"
	"io/ioutil"

	"github.com/sirupsen/logrus"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// ContainerdLogger adapts errnie's logging system to containerd's logger interface
type ContainerdLogger struct {
	*logrus.Logger
}

func NewContainerdLogger() *logrus.Logger {
	logger := logrus.New()

	// Configure the logger to use errnie's handlers
	logger.AddHook(&errnieHook{})

	// Disable the default output since we're using errnie
	logger.SetOutput(ioutil.Discard)

	// Set the formatter to JSON to match containerd's default
	logger.SetFormatter(&logrus.JSONFormatter{})

	return logger
}

// errnieHook implements logrus.Hook to redirect logs to errnie
type errnieHook struct{}

func (h *errnieHook) Levels() []logrus.Level {
	return logrus.AllLevels
}

func (h *errnieHook) Fire(entry *logrus.Entry) error {
	// Format message with fields
	msg := entry.Message
	if len(entry.Data) > 0 {
		fields := make([]string, 0, len(entry.Data))
		for k, v := range entry.Data {
			fields = append(fields, fmt.Sprintf("%s=%v", k, v))
		}
		msg = fmt.Sprintf("%s [%s]", msg, fields)
	}

	switch entry.Level {
	case logrus.DebugLevel, logrus.TraceLevel:
		errnie.Debug(msg)
	case logrus.InfoLevel:
		errnie.Info(msg)
	case logrus.WarnLevel:
		errnie.Warn(msg)
	case logrus.ErrorLevel, logrus.FatalLevel, logrus.PanicLevel:
		errnie.Error(msg)
	}
	return nil
}

func (l *ContainerdLogger) Info(msg string, fields ...interface{}) {
	errnie.Info(fmt.Sprintf(msg, fields...))
}

func (l *ContainerdLogger) Warn(msg string, fields ...interface{}) {
	errnie.Warn(fmt.Sprintf(msg, fields...))
}

func (l *ContainerdLogger) Error(msg string, fields ...interface{}) {
	errnie.Error(fmt.Sprintf(msg, fields...))
}

func (l *ContainerdLogger) Debug(msg string, fields ...interface{}) {
	errnie.Debug(fmt.Sprintf(msg, fields...))
}

func (l *ContainerdLogger) Trace(msg string, fields ...interface{}) {
	errnie.Debug(fmt.Sprintf(msg, fields...)) // Using Debug since errnie doesn't have Trace
}

func (l *ContainerdLogger) WithField(key string, value interface{}) *logrus.Entry {
	return l.WithField(key, value)
}

func (l *ContainerdLogger) WithFields(fields map[string]interface{}) *logrus.Entry {
	return l.WithFields(fields)
}

func (l *ContainerdLogger) WithError(err error) *logrus.Entry {
	return l.WithError(err)
}
