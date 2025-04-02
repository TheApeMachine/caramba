package core

type Status uint

const (
	StatusUnknown Status = iota
	StatusReady
	StatusWaiting
	StatusWorking
	StatusDone
	StatusError
)
