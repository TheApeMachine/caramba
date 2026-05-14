package validate

import (
	"errors"
	"maps"
	"reflect"
	"slices"
)

/*
missingDependency reports whether a value passed as a required dependency
is absent. A nil interface value is absent. So is a typed nil pointer, map,
slice, channel, or func stored in an any slot — the usual Go interface nil
trap — because those values cannot be used safely without checking Kind and
IsNil first.
*/
func missingDependency(obj any) bool {
	if obj == nil {
		return true
	}

	value := reflect.ValueOf(obj)

	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice, reflect.UnsafePointer:
		return value.IsNil()
	default:
		return false
	}
}

/*
Require validates that all required dependencies are present (see
missingDependency). Keys are checked in sorted order so the reported name
is stable when several entries are wrong.

Pass a map of name → value; if any dependency is missing, returns an error
with a clear message (e.g. "pool is required"). Use in constructors after
options are applied so callers fail fast instead of ad-hoc nil checks
throughout Run() and other methods.
*/
func Require(objs map[string]any) error {
	names := slices.Collect(maps.Keys(objs))
	slices.Sort(names)

	for _, name := range names {
		if missingDependency(objs[name]) {
			return errors.New(name + " is required")
		}
	}

	return nil
}
