package provider

import "context"

/*
Provider is an interface that objects can implement if they want to
present themselves as an LLM provider.
*/
type Provider interface {
	Generate(context.Context, *GenerationParams) <-chan Event
}
