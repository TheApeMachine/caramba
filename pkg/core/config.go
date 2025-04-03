package core

import (
	"fmt"
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

var (
	once sync.Once
	cfg  *Config
)

type Config struct {
	OpenAIAPIKey string                 `yaml:"openai_api_key"`
	Protocols    map[string]RawProtocol `yaml:"protocols"`
}

// RawProtocol represents the protocol as defined in the YAML file
type RawProtocol struct {
	Steps []string `yaml:"steps"`
}

type ConfigOption func(*Config)

func NewConfig(opts ...ConfigOption) *Config {
	once.Do(func() {
		cfg = &Config{
			Protocols: make(map[string]RawProtocol),
		}

		for _, opt := range opts {
			opt(cfg)
		}
	})

	return cfg
}

// LoadFromFile loads configuration from a YAML file
func (c *Config) LoadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("error reading config file: %w", err)
	}

	// Replace environment variables
	data = []byte(os.ExpandEnv(string(data)))

	if err := yaml.Unmarshal(data, c); err != nil {
		return fmt.Errorf("error parsing config file: %w", err)
	}

	return nil
}

func WithOpenAIAPIKey(key string) ConfigOption {
	return func(c *Config) {
		c.OpenAIAPIKey = key
	}
}

func WithProtocol(name string, protocol RawProtocol) ConfigOption {
	return func(c *Config) {
		c.Protocols[name] = protocol
	}
}
