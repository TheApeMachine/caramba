package tool

import (
	"errors"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (tool *Tool) Is(state errnie.State) (ok bool) {
	ok = tool.State() == uint64(state)
	errnie.Trace("tool.Is", "id", tool.ID(), "state", state, "ok", ok)
	return
}

func (tool *Tool) ToState(state errnie.State) *Tool {
	errnie.Trace("tool.ToState", "id", tool.ID(), "state", state)
	tool.SetState(uint64(state))
	errnie.Debug("tool.ToState", "id", tool.ID(), "state", state)
	return tool
}

func (tool *Tool) ID() string {
	errnie.Trace("tool.ID", "id", errnie.Try(tool.Uuid()))
	uid := errnie.Try(tool.Uuid())

	if uid == "" {
		if errnie.Error(tool.SetUuid(uuid.New().String())) != nil {
			errnie.Fatal(errnie.New(
				errnie.WithError(errors.New("failed to set uuid")),
				errnie.WithMessage("failed to set uuid"),
			))
		}
	}

	return uid
}
