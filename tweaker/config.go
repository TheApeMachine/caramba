package tweaker

import (
	"fmt"
	"sync"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/utils"
)

/*
Config provides a thread-safe way to access configuration values,
with support for round-robin access to repeated keys.
*/
type Config struct {
	baseKey string
	cache   map[string]*Capsule[string]
	mu      sync.RWMutex
}

func NewConfig(baseKey string) *Config {
	return &Config{
		baseKey: baseKey,
		cache:   make(map[string]*Capsule[string]),
	}
}

/*
S is a shorthand to get a string value from the config. It takes a variable
number of keys, where it will use the first key normally, if there are no
other keys. If there are other keys, each time that S is called with the
same key sequence, it will round-robin through the keys, and return the
value of the next key in the sequence. This is very useful for creating
dynamically sequenced prompts.

For example:

key0: Returns a prompt which describes some task.
key1: Returns a prompt which instructs the model to review the previous response.
key2: Returns a prompt which instructs the model to improve the response.
*/
func (cfg *Config) S(keys ...string) string {
	if len(keys) == 0 {
		return ""
	}

	// Create a unique cache key for this sequence of keys
	cacheKey := utils.JoinWith(".", append([]string{cfg.baseKey}, keys...)...)

	cfg.mu.RLock()
	capsule, exists := cfg.cache[cacheKey]
	cfg.mu.RUnlock()

	if !exists {
		var values []string
		for _, key := range keys {
			fullKey := fmt.Sprintf("%s.%s", cfg.baseKey, key)
			if val := viper.GetString(fullKey); val != "" {
				values = append(values, val)
			}
		}

		capsule = NewCapsule(values)

		cfg.mu.Lock()
		cfg.cache[cacheKey] = capsule
		cfg.mu.Unlock()
	}

	return capsule.Next()
}
