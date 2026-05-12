package config

import (
	"os"

	"github.com/spf13/viper"
)

func WithDefault[T any](key string, defaultValue T) T {
	v := viper.GetViper()

	expandIfString := func(value T) T {
		asString, ok := any(value).(string)
		if !ok {
			return value
		}

		return any(os.ExpandEnv(asString)).(T)
	}

	if !v.IsSet(key) {
		return expandIfString(defaultValue)
	}

	raw := v.Get(key)

	if raw == nil {
		return expandIfString(defaultValue)
	}

	value, ok := raw.(T)

	if !ok {
		return expandIfString(defaultValue)
	}

	return expandIfString(value)
}
