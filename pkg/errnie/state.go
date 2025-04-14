package errnie

type StateType int

type State struct {
	current StateType
	history []StateType
}

const (
	StateUnknown StateType = iota
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

func NewState() *State {
	return &State{
		current: StateInitialized,
		history: make([]StateType, 0),
	}
}

func (state *State) To(s StateType) *State {
	state.history = append(state.history, state.current)
	state.current = s
	return state
}

func (state *State) IsReady() bool  { return state.current == StateReady }
func (state *State) IsBusy() bool   { return state.current == StateBusy }
func (state *State) IsDone() bool   { return state.current == StateDone }
func (state *State) IsFailed() bool { return state.current == StateFailed }
func (state *State) IsError() bool  { return state.current == StateError }

func (state *State) String() string {
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
	}[state.current]
}
