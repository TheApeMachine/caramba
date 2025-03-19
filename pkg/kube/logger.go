package kube

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/errnie"
	"sigs.k8s.io/kind/pkg/log"
)

// KindLogger adapts errnie's logging system to Kind's logger interface
type KindLogger struct{}

func NewKindLogger() log.Logger {
	return &KindLogger{}
}

func (l *KindLogger) Info(message string) {
	errnie.Info(message)
}

func (l *KindLogger) Infof(format string, args ...interface{}) {
	errnie.Info(fmt.Sprintf(format, args...))
}

func (l *KindLogger) Warn(message string) {
	errnie.Warn(message)
}

func (l *KindLogger) Warnf(format string, args ...interface{}) {
	errnie.Warn(fmt.Sprintf(format, args...))
}

func (l *KindLogger) Error(message string) {
	errnie.Error(message)
}

func (l *KindLogger) Errorf(format string, args ...interface{}) {
	errnie.Error(fmt.Sprintf(format, args...))
}

func (l *KindLogger) V(level log.Level) log.InfoLogger {
	return l // Always return self since errnie handles levels internally
}

func (l *KindLogger) Enabled() bool {
	return true // Always enabled since errnie handles its own levels
}
