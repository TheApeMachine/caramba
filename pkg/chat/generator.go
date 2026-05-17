package chat

import (
	"context"
	"io"
)

/*
Generator streams an assistant response for a prompt.
*/
type Generator interface {
	Generate(ctx context.Context, prompt string, emit func(string) error) error
}

/*
SessionRunner is the optional interface a Generator can implement to
take ownership of the whole interactive session — banner, per-turn
user prompt, command handling, decode loop. Session.Run prefers this
path when the underlying generator exposes it because the runtime
manifest is the authoritative source of session behavior; the legacy
"prompt-loop in Go" path remains available for generators (like the
preview generator) that do not have a runtime program of their own.
*/
type SessionRunner interface {
	RunSession(ctx context.Context, input io.Reader, output io.Writer) error
}
