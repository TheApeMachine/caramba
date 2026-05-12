package qpool

/*
Evidence optionally records weighted hints supporting a quantum-style state's likelihood.

Instances are typically appended to State.Evidence when a caller wants to attach diagnostic scores without replacing the primary Value payload.
*/
type Evidence struct {
	Source string
	Weight float64
}

/*
State models one branch of a superposed value inside qpool's quantum-inspired structures: a concrete payload (Value) plus optional amplitude, probability mass, and supporting evidence slices.

Value holds the collapsed or candidate payload; it may be nil when the state only carries weights. Probability is a normalized or heuristic likelihood in [0,1] when populated; Amplitude is a complex weight used by algorithms that combine interfered branches; Evidence lists auxiliary measurements.

Constructors and observers in this package read these fields when collapsing or comparing states; mutators should keep Probability consistent with Amplitude when both are used together.
*/
type State struct {
	Value       any
	Probability float64
	Amplitude   complex128
	Evidence    []Evidence
}
