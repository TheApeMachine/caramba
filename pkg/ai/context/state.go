package context

import (
	"errors"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (ctx Context) Is(state errnie.State) (ok bool) {
	ok = ctx.State() == uint64(state)
	errnie.Trace("context.Is", "id", ctx.ID(), "state", state, "ok", ok)
	return
}

func (ctx Context) ToState(state errnie.State) Context {
	errnie.Trace("context.ToState", "id", ctx.ID(), "state", state)
	ctx.SetState(uint64(state))
	errnie.Debug("context.ToState", "id", ctx.ID(), "state", state)
	return ctx
}

func (ctx Context) ID() string {
	errnie.Trace("context.ID", "id", errnie.Try(ctx.Uuid()))
	uid := errnie.Try(ctx.Uuid())

	if uid == "" {
		if errnie.Error(ctx.SetUuid(uuid.New().String())) != nil {
			errnie.Fatal(errnie.New(
				errnie.WithError(errors.New("failed to set uuid")),
				errnie.WithMessage("failed to set uuid"),
			))
		}
	}

	return uid
}
