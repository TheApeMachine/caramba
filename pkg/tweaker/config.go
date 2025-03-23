package tweaker

import (
	"fmt"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var (
	cfg        *Config
	defaultKey = "settings.defaults."
)

func init() {
	cfg = NewConfig()
}

type Config struct {
	v func() *viper.Viper
}

func NewConfig() *Config {
	return &Config{
		v: func() *viper.Viper {
			return viper.GetViper()
		},
	}
}

// get is a generic helper function to retrieve values from viper with error handling
func get[T any](key string, getter func(string) T, defaultValue T) T {
	if out := getter(key); !isEmpty(out) {
		return out
	}
	errnie.Error(fmt.Errorf("%s not set", key))
	return defaultValue
}

// isEmpty is a helper to check for zero values of different types
func isEmpty(v any) bool {
	switch val := v.(type) {
	case string:
		return val == ""
	case int:
		return val == 0
	case float64:
		return val == 0
	case []string:
		return len(val) == 0
	case bool:
		return !val
	default:
		return false
	}
}

func GetModel(provider string) string {
	return get("models."+provider, cfg.v().GetString, "")
}

func GetEndpoint(provider string) string {
	return get("endpoints."+provider, cfg.v().GetString, "")
}

func GetProvider() string {
	return get(defaultKey+"provider", cfg.v().GetString, "")
}

func GetSystemPrompt(key string) string {
	return get(defaultKey+"system_prompt."+key, cfg.v().GetString, "")
}

func GetTemperature() float64 {
	return get(defaultKey+"temperature", cfg.v().GetFloat64, 0.0)
}

func GetTopP() float64 {
	return get(defaultKey+"top_p", cfg.v().GetFloat64, 0.0)
}

func GetTopK() int {
	return get(defaultKey+"top_k", cfg.v().GetInt, 0)
}

func GetPresencePenalty() float64 {
	return get(defaultKey+"presence_penalty", cfg.v().GetFloat64, 0.0)
}

func GetFrequencyPenalty() float64 {
	return get(defaultKey+"frequency_penalty", cfg.v().GetFloat64, 0.0)
}

func GetMaxTokens() int {
	return get(defaultKey+"max_tokens", cfg.v().GetInt, 0)
}

func GetStopSequences() []string {
	return get(defaultKey+"stop_sequences", cfg.v().GetStringSlice, []string{})
}

func GetStream() bool {
	return get(defaultKey+"stream", cfg.v().GetBool, false)
}

func GetOS() string {
	return get(defaultKey+"platform.os", cfg.v().GetString, "")
}

func GetArch() string {
	return get(defaultKey+"platform.arch", cfg.v().GetString, "")
}

func GetVariant() string {
	return get(defaultKey+"platform.variant", cfg.v().GetString, "")
}
