package agent

import "github.com/theapemachine/caramba/pkg/errnie"

func (agent Agent) Is(state errnie.State) bool {
	return agent.State() == uint64(state)
}

func (agent Agent) ToState(state errnie.State) Agent {
	agent.SetState(uint64(state))
	return agent
}

func (agent Agent) ID() string {
	return errnie.Try(errnie.Try(agent.Identity()).Identifier())
}
