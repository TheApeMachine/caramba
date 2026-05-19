package dropout

/*
DropoutConfig carries the drop rate and deterministic seed for
inverted dropout scaling.
*/
type DropoutConfig struct {
	Rate float32
	Seed uint64
}

func DefaultDropoutConfig() DropoutConfig {
	return DropoutConfig{Rate: 0.1, Seed: 0xc0ffee}
}
