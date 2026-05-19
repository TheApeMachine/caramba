package sampling

type SamplingConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
	Seed        uint64
}

func DefaultSamplingConfig() SamplingConfig {
	return SamplingConfig{Temperature: 1.0, TopK: 0, TopP: 1.0, Seed: 0xfeedface}
}
