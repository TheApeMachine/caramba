package rope

type RoPEConfig struct {
	BaseFreq      float64
	StartPosition int
}

func DefaultRoPEConfig() RoPEConfig {
	return RoPEConfig{BaseFreq: 10000.0, StartPosition: 0}
}
