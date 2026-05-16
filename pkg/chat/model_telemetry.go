package chat

import (
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/qpool"
)

const chatTelemetryComponent = "chat"

/*
modelStartupTelemetry publishes model startup milestones through qpool.
It keeps the CLI visible while Hub assets and serialized weights resolve.
*/
type modelStartupTelemetry struct {
	manifest  string
	model     string
	tokenizer string
	backend   string
}

/*
newModelStartupTelemetry creates startup telemetry scoped to one model load.
*/
func newModelStartupTelemetry(config ModelConfig) *modelStartupTelemetry {
	telemetry := &modelStartupTelemetry{}
	telemetry.Apply(config)

	return telemetry
}

/*
Apply refreshes the fields that are known after manifest runtime resolution.
*/
func (telemetry *modelStartupTelemetry) Apply(config ModelConfig) {
	if telemetry == nil {
		return
	}

	telemetry.manifest = strings.TrimSpace(config.Manifest)
	telemetry.model = strings.TrimSpace(config.Model)
	telemetry.tokenizer = strings.TrimSpace(config.Tokenizer)
	telemetry.backend = strings.TrimSpace(config.Backend)
}

/*
SetBackend records the concrete backend location once selected.
*/
func (telemetry *modelStartupTelemetry) SetBackend(backend string) {
	if telemetry == nil {
		return
	}

	telemetry.backend = strings.TrimSpace(backend)
}

/*
Publish emits an info event for a startup stage.
*/
func (telemetry *modelStartupTelemetry) Publish(
	op string,
	message string,
	fields ...qpool.Field,
) {
	if telemetry == nil {
		return
	}

	event := qpool.NewInfoEvent(
		chatTelemetryComponent,
		op,
		message,
		append(telemetry.fields(), fields...),
	)
	event.WithTime(time.Now())
	qpool.Publish(event)
}

/*
Error emits an error event and returns the original error.
*/
func (telemetry *modelStartupTelemetry) Error(
	op string,
	message string,
	err error,
	fields ...qpool.Field,
) error {
	if telemetry == nil || err == nil {
		return err
	}

	event := qpool.NewErrorEvent(
		chatTelemetryComponent,
		op,
		message,
		err,
		append(telemetry.fields(), fields...),
	)
	event.WithTime(time.Now())
	qpool.Publish(event)

	return err
}

func (telemetry *modelStartupTelemetry) fields() []qpool.Field {
	fields := []qpool.Field{
		{Key: "manifest", Value: telemetry.manifest},
	}

	if telemetry.model != "" {
		fields = append(fields, qpool.Field{Key: "model", Value: telemetry.model})
	}

	if telemetry.tokenizer != "" {
		fields = append(fields, qpool.Field{Key: "tokenizer", Value: telemetry.tokenizer})
	}

	if telemetry.backend != "" {
		fields = append(fields, qpool.Field{Key: "backend", Value: telemetry.backend})
	}

	return fields
}
