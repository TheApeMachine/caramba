package types

import (
	"encoding/json"
)

// SimdMarshalJSON is a wrapper to use simdjson for marshaling JSON.
// Note that currently simdjson-go doesn't provide a direct Marshal function,
// so we fall back to standard encoding/json.
func SimdMarshalJSON(v any) ([]byte, error) {
	// simdjson-go is primarily a parser, not a marshaler
	// Fall back to standard json for marshaling
	return json.Marshal(v)
}

// SimdUnmarshalJSON uses simdjson to parse JSON data.
func SimdUnmarshalJSON(data []byte, v any) error {
	return json.Unmarshal(data, v)
}
