package executor

import "fmt"

/*
Scope is the lexically scoped local variable table. The executor
pushes a frame when entering a body and pops it on exit so loop
bindings (control.loop_each `as: timestep`) shadow outer values
without leaking.
*/
type Scope struct {
	parent *Scope
	frame  map[string]any
}

/*
NewScope creates a root scope with no parent.
*/
func NewScope() *Scope {
	return &Scope{frame: map[string]any{}}
}

/*
Child returns a new scope whose parent is the receiver. Lookups
bubble up; writes always land on the child's own frame.
*/
func (scope *Scope) Child() *Scope {
	return &Scope{parent: scope, frame: map[string]any{}}
}

/*
Get walks the parent chain until the name is found.
*/
func (scope *Scope) Get(name string) (any, error) {
	for cursor := scope; cursor != nil; cursor = cursor.parent {
		if value, ok := cursor.frame[name]; ok {
			return value, nil
		}
	}

	return nil, fmt.Errorf("runtime/executor: local %q is not bound", name)
}

/*
Set writes the value into the current frame, regardless of whether a
parent binding exists. This matches the program contract where each
step's output references explicitly target a local name.
*/
func (scope *Scope) Set(name string, value any) {
	scope.frame[name] = value
}

/*
Has reports whether the name resolves in this scope or any ancestor.
*/
func (scope *Scope) Has(name string) bool {
	for cursor := scope; cursor != nil; cursor = cursor.parent {
		if _, ok := cursor.frame[name]; ok {
			return true
		}
	}

	return false
}
