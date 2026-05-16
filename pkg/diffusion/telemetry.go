package diffusion

import (
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/qpool"
)

const telemetryComponent = "diffusion"

type Telemetry struct {
	manifest string
	model    string
	backend  string
}

func NewTelemetry(config Config) *Telemetry {
	telemetry := &Telemetry{}
	telemetry.Apply(config)

	return telemetry
}

func (telemetry *Telemetry) Apply(config Config) {
	if telemetry == nil {
		return
	}

	telemetry.manifest = strings.TrimSpace(config.Manifest)
	telemetry.model = strings.TrimSpace(config.Model.Source)
	telemetry.backend = strings.TrimSpace(config.Backend)
}

func (telemetry *Telemetry) SetBackend(backend string) {
	if telemetry == nil {
		return
	}

	telemetry.backend = strings.TrimSpace(backend)
}

func (telemetry *Telemetry) Publish(op string, message string, fields ...qpool.Field) {
	if telemetry == nil {
		return
	}

	event := qpool.NewInfoEvent(
		telemetryComponent,
		op,
		message,
		append(telemetry.fields(), fields...),
	)
	event.WithTime(time.Now())
	qpool.Publish(event)
}

func (telemetry *Telemetry) Error(
	op string,
	message string,
	err error,
	fields ...qpool.Field,
) error {
	if telemetry == nil || err == nil {
		return err
	}

	event := qpool.NewErrorEvent(
		telemetryComponent,
		op,
		message,
		err,
		append(telemetry.fields(), fields...),
	)
	event.WithTime(time.Now())
	qpool.Publish(event)

	return err
}

func (telemetry *Telemetry) fields() []qpool.Field {
	fields := []qpool.Field{{Key: "manifest", Value: telemetry.manifest}}

	if telemetry.model != "" {
		fields = append(fields, qpool.Field{Key: "model", Value: telemetry.model})
	}

	if telemetry.backend != "" {
		fields = append(fields, qpool.Field{Key: "backend", Value: telemetry.backend})
	}

	return fields
}
