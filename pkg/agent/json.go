package agent

import (
	"encoding/json"

	"github.com/minio/simdjson-go"
)

// SimdMarshalJSON is a wrapper to use simdjson for marshaling JSON.
// Note that currently simdjson-go doesn't provide a direct Marshal function,
// so we fall back to standard encoding/json.
func SimdMarshalJSON(v interface{}) ([]byte, error) {
	// simdjson-go is primarily a parser, not a marshaler
	// Fall back to standard json for marshaling
	return json.Marshal(v)
}

// SimdUnmarshalJSON uses simdjson to parse JSON data.
func SimdUnmarshalJSON(data []byte, v interface{}) error {
	// Parse the JSON
	pj, err := simdjson.Parse(data, nil)
	if err != nil {
		return err
	}

	// Get the iterator
	iter := pj.Iter()
	iter.AdvanceInto()

	// Convert to Go interface types
	iface, err := iter.Interface()
	if err != nil {
		return err
	}

	// Convert to JSON and back to fill the target struct
	// This is necessary because simdjson-go doesn't provide direct struct unmarshaling
	jsonData, err := json.Marshal(iface)
	if err != nil {
		return err
	}

	return json.Unmarshal(jsonData, v)
}
