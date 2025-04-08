package errnie

type State uint64

const (
	StateUnknown State = iota
	StateInitialized
	StateIdle
	StatePending
	StateWaiting
	StateReady
	StateBusy
	StateDone
	StateFailed
	StateError
)

func (state State) String() string {
	return []string{
		"unknown",
		"initialized",
		"idle",
		"pending",
		"waiting",
		"ready",
		"busy",
		"done",
		"failed",
		"error",
	}[state]
}
