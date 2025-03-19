package core

import (
	"sync"
)

var (
	once sync.Once
	cfg  *Config
)

type Config struct {
	OpenAIAPIKey string `yaml:"openai_api_key"`
}

type ConfigOption func(*Config)

func NewConfig(opts ...ConfigOption) *Config {
	once.Do(func() {
		cfg = &Config{}

		for _, opt := range opts {
			opt(cfg)
		}
	})

	return cfg
}

func WithOpenAIAPIKey(key string) ConfigOption {
	return func(c *Config) {
		c.OpenAIAPIKey = key
	}
}
