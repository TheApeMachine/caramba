package errnie

type Return struct {
	Value any
	Error error
}

// JumpReturn triggers a non-local return when there's an error
func JumpReturn[T any](v T, err error) T {
	if err != nil {
		panic(Return{Value: v, Error: err})
	}
	return v
}

// Safe wraps a function call that might return an error
// If there's an error, it will be caught and returned
// If there's no error, the value is returned
func Safe[T any](fn func() T) (result T) {
	defer func() {
		if r := recover(); r != nil {
			if ret, ok := r.(Return); ok {
				result = ret.Value.(T)
			} else {
				panic(r) // Re-panic if it's not our error type
			}
		}
	}()

	return fn()
}

// Try is a helper that makes it cleaner to use Safe with functions that return (T, error)
func Try[T any](v T, err error) T {
	return Safe(func() T {
		return JumpReturn(v, err)
	})
}
