package datura

import (
	"fmt"
	"strconv"

	"github.com/theapemachine/caramba/pkg/errnie"
)

// GetMetaValue retrieves a typed metadata value from an artifact
func GetMetaValue[T any](artifact *Artifact, key string) T {
	var (
		metadata Artifact_Metadata_List
		err      error
		result   T
	)

	if metadata, err = artifact.Metadata(); errnie.Error(err) != nil {
		return *new(T)
	}

	for idx := range metadata.Len() {
		var (
			k    string
			meta = metadata.At(idx)
		)

		if k, err = meta.Key(); errnie.Error(err) != nil {
			return *new(T)
		}

		if k == key {
			which := meta.Value().Which()

			// Use type assertion to determine the return type expected
			switch any(result).(type) {
			case string:
				// Handle string type
				if which == Artifact_Metadata_value_Which_textValue {
					val, _ := meta.Value().TextValue()
					return any(val).(T)
				}
			case int:
				// Handle integer type
				if which == Artifact_Metadata_value_Which_intValue {
					val := meta.Value().IntValue()
					return any(int(val)).(T)
				} else if which == Artifact_Metadata_value_Which_textValue {
					// Try to convert from string
					val, _ := meta.Value().TextValue()
					if i, err := strconv.Atoi(val); err == nil {
						return any(i).(T)
					}
				}
			case float64:
				// Handle float type
				if which == Artifact_Metadata_value_Which_floatValue {
					val := meta.Value().FloatValue()
					return any(val).(T)
				} else if which == Artifact_Metadata_value_Which_textValue {
					// Try to convert from string
					val, _ := meta.Value().TextValue()
					if f, err := strconv.ParseFloat(val, 64); err == nil {
						return any(f).(T)
					}
				}
			case bool:
				// Handle boolean type
				if which == Artifact_Metadata_value_Which_boolValue {
					val := meta.Value().BoolValue()
					return any(val).(T)
				} else if which == Artifact_Metadata_value_Which_textValue {
					// Try to convert from string
					val, _ := meta.Value().TextValue()
					if b, err := strconv.ParseBool(val); err == nil {
						return any(b).(T)
					}
				}
			}
		}
	}

	errnie.Trace("datura.GetMetaValue", "key", key, "value", result)

	return *new(T)
}

// SetMetaValue sets a metadata value with the appropriate type
func (artifact *Artifact) SetMetaValue(key string, val any) *Artifact {
	errnie.Trace("datura.SetMetaValue", "key", key, "value", val)

	// Create a new option function
	setOption := func(builder *Artifact) {
		var (
			mdList    Artifact_Metadata_List
			newMdList Artifact_Metadata_List
			err       error
		)

		// Get existing metadata
		if mdList, err = builder.Metadata(); errnie.Error(err) != nil {
			return
		}

		// Create new metadata list with space for one more item
		if newMdList, err = builder.NewMetadata(
			int32(mdList.Len() + 1),
		); errnie.Error(err) != nil {
			return
		}

		// Copy existing metadata
		for idx := range mdList.Len() {
			if err = newMdList.Set(idx, mdList.At(idx)); errnie.Error(err) != nil {
				return
			}
		}

		// Add the new item
		item := newMdList.At(newMdList.Len() - 1)
		if err = item.SetKey(key); errnie.Error(err) != nil {
			return
		}

		// Set value based on type
		switch v := val.(type) {
		case string:
			item.Value().SetTextValue(v)
		case int:
			item.Value().SetIntValue(int64(v))
		case int64:
			item.Value().SetIntValue(v)
		case float64:
			item.Value().SetFloatValue(v)
		case bool:
			item.Value().SetBoolValue(v)
		case []byte:
			item.Value().SetBinaryValue(v)
		default:
			// Default to string representation
			item.Value().SetTextValue(fmt.Sprintf("%v", v))
		}
	}

	// Apply the option
	setOption(artifact)
	return artifact
}
