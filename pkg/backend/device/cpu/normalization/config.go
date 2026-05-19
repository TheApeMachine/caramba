package normalization

type GroupNormConfig struct {
	Groups int
}

func DefaultGroupNormConfig() GroupNormConfig {
	return GroupNormConfig{Groups: 32}
}
