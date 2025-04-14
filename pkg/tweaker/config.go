package tweaker

import (
	"fmt"
	"strings"

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
	errnie.Debug("tweaker.NewConfig")

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
	errnie.New(errnie.WithError(fmt.Errorf("%s not set", key)))
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
	return cfg.v().GetFloat64("settings.defaults.presence_penalty")
}

func GetFrequencyPenalty() float64 {
	return cfg.v().GetFloat64("settings.defaults.frequency_penalty")
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

func GetQdrantCollection() string {
	return get("tools.qdrant.collection", cfg.v().GetString, "")
}

func GetQdrantDimension() int64 {
	return get("tools.qdrant.dimension", cfg.v().GetInt64, 0)
}

func GetQdrantHost() string {
	return get("tools.qdrant.host", cfg.v().GetString, "")
}

func GetQdrantPort() int {
	return get("tools.qdrant.port", cfg.v().GetInt, 0)
}

func GetQdrantAPIKey() string {
	return get("tools.qdrant.api_key", cfg.v().GetString, "")
}

func WithVariables(
	key string,
	variable1 string, value1 string,
	variable2 string, value2 string,
) string {
	return strings.ReplaceAll(
		strings.ReplaceAll(
			cfg.v().GetString(key),
			"{{"+variable1+"}}", value1,
		),
		"{{"+variable2+"}}", value2,
	)
}

func WithVariable(key string, variable string, value string) string {
	return strings.ReplaceAll(cfg.v().GetString(key), "{{"+variable+"}}", value)
}

// Get returns a string value from the config with a default value
func Get(key string, defaultValue string) string {
	return get(key, cfg.v().GetString, defaultValue)
}

// GetStringMap returns a map[string]any from the config
func GetStringMap(key string) map[string]any {
	return cfg.v().GetStringMap(key)
}

// GetStringSlice returns a []string from the config
func GetStringSlice(key string) []string {
	return cfg.v().GetStringSlice(key)
}

func GetProtocol(key string) []string {
	return get("protocols."+key, cfg.v().GetStringSlice, []string{})
}

func Value[T any](key string) T {
	val := cfg.v().Get(key)

	switch v := val.(type) {
	case T:
		return v
	default:
		return val.(T)
	}
}
