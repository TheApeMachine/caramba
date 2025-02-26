/*
Package util provides utility functions for the agent system.
This file contains utilities for JSON schema generation used with OpenAI's structured outputs.
*/
package util

import (
	"github.com/invopop/jsonschema"
)

// GenerateSchema is a helper function that creates a JSON schema from a type
// This version is compatible with OpenAI's structured outputs feature
func GenerateSchema[T any]() interface{} {
	// Structured Outputs uses a subset of JSON schema
	// These flags are necessary to comply with the subset
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties:  false,
		DoNotReference:             true,
		RequiredFromJSONSchemaTags: true,
		ExpandedStruct:             true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
}
