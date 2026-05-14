package state

/*
Operation applies one compute operation to a state dictionary.
*/
type Operation interface {
	Forward(*Dict) (*Dict, error)
}
