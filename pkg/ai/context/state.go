package context

import "github.com/theapemachine/caramba/pkg/errnie"

func (ctx Context) Is(state errnie.State) bool {
	return ctx.State() == uint64(state)
}

func (ctx Context) ToState(state errnie.State) Context {
	ctx.SetState(uint64(state))
	return ctx
}

func (ctx Context) ID() string {
	return errnie.Try(ctx.Uuid())
}
