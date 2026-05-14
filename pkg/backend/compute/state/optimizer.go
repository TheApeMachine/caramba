package state

/*
Optimizer applies one optimizer step to a state dictionary.
*/
type Optimizer interface {
	Step(*Dict) (*Dict, error)
}
