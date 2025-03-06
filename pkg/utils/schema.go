package utils

import (
	"github.com/invopop/jsonschema"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func GenerateSchema[T any]() any {
	errnie.Debug("GenerateSchema")

	// Structured Outputs uses a subset of JSON schema
	// These flags are necessary to comply with the subset
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
}
