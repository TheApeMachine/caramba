package chat

import "context"

/*
Generator streams an assistant response for a prompt.
*/
type Generator interface {
	Generate(ctx context.Context, prompt string, emit func(string) error) error
}
