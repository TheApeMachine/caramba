package environment

// Config holds runtime configuration
type Config struct {
	RuntimeType string // "docker" or "caramba"
}

// NewRuntime creates a new container runtime based on the config
func NewRuntime(cfg Config) (Runtime, error) {
	switch cfg.RuntimeType {
	case "docker":
		return newDockerRuntime()
	default:
		return newDockerRuntime() // default to Docker for now
	}
}
