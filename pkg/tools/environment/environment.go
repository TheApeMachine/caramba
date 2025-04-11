package environment

import (
	fs "github.com/theapemachine/caramba/pkg/stores/fs"
)

// Config holds runtime configuration
type Config struct {
	RuntimeType string    // "docker" or "caramba"
	FSStore     *fs.Store // Added filesystem store
}

// NewRuntime creates a new container runtime based on the config
func NewRuntime(cfg Config) (Runtime, error) {
	return newDockerRuntime(cfg.FSStore)
}
