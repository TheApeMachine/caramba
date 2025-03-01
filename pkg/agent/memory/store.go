package memory

import "context"

/*
Store is an interface that objects must implement if they want to act as
a memory store for Caramba agents.
*/
type Store interface {
	Query(context.Context, map[string]any) (string, error)
	Mutate(context.Context, map[string]any) error
}
