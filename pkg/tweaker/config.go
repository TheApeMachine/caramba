package tweaker

import "github.com/spf13/viper"

var (
	cfg *Config
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

func GetModel(provider string) string {
	return cfg.v().GetString("models." + provider)
}

func GetEndpoint(provider string) string {
	return cfg.v().GetString("endpoints." + provider)
}

func GetProvider() string {
	return cfg.v().GetString("settings.defaults.provider")
}

func GetSystemPrompt() string {
	return cfg.v().GetString("settings.defaults.system_prompt")
}

func GetTemperature() float64 {
	return cfg.v().GetFloat64("settings.defaults.temperature")
}

func GetTopP() float64 {
	return cfg.v().GetFloat64("settings.defaults.top_p")
}

func GetTopK() int {
	return cfg.v().GetInt("settings.defaults.top_k")
}

func GetPresencePenalty() float64 {
	return cfg.v().GetFloat64("settings.defaults.presence_penalty")
}

func GetFrequencyPenalty() float64 {
	return cfg.v().GetFloat64("settings.defaults.frequency_penalty")
}

func GetMaxTokens() int {
	return cfg.v().GetInt("settings.defaults.max_tokens")
}

func GetStopSequences() []string {
	return cfg.v().GetStringSlice("settings.defaults.stop_sequences")
}

func GetStream() bool {
	return false
}
