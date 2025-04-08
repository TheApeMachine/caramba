package message

import "github.com/theapemachine/caramba/pkg/errnie"

func (msg Message) Is(state errnie.State) bool {
	return msg.State() == uint64(state)
}

func (msg Message) ToState(state errnie.State) Message {
	msg.SetState(uint64(state))
	return msg
}

func (msg Message) ID() string {
	return errnie.Try(msg.Uuid())
}
