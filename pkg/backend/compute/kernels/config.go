package kernels

/*
DropoutConfig carries deterministic dropout parameters shared by tests and
device implementations.
*/
type DropoutConfig struct {
	Rate float32
	Seed uint64
}

/*
DefaultDropoutConfig returns the canonical dropout configuration.
*/
func DefaultDropoutConfig() DropoutConfig {
	return DropoutConfig{Rate: 0.1, Seed: 0xc0ffee}
}

/*
SamplingConfig carries deterministic top-k/top-p sampling parameters.
*/
type SamplingConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
	Seed        uint64
}

/*
DefaultSamplingConfig returns the canonical sampling configuration.
*/
func DefaultSamplingConfig() SamplingConfig {
	return SamplingConfig{Temperature: 1.0, TopK: 0, TopP: 1.0, Seed: 0xfeedface}
}
