package errnie

// Return struct now holds *ErrnieError
type Return struct {
	Value any
	Error *ErrnieError // Changed from error
}

// In pkg/errnie/safe.go (or a new file)

// RunSafely executes the given function and recovers from any *ErrnieError panic
// triggered by functions like errnie.Try within fn.
// It returns the recovered *ErrnieError as a standard error, or nil if no panic occurred.
func RunSafely(fn func()) (err error) {
    defer func() {
        if r := recover(); r != nil {
            // Check if the panic was caused by an *ErrnieError
            if ee, ok := r.(*ErrnieError); ok {
                // Return the ErrnieError as a standard error
                err = ee
            } else {
                // It was a different panic, re-panic
                panic(r)
            }
        }
    }()

    // Execute the user's function
    fn()
    return nil // No panic occurred
}

// Optional version if the wrapped function itself needs to return an error
func RunSafelyErr(fn func() error) (err error) {
    defer func() {
        if r := recover(); r != nil {
            if ee, ok := r.(*ErrnieError); ok {
                err = ee // Prioritize the panic error
            } else {
                panic(r)
            }
        }
    }()

    // Execute the user's function and capture its return error
    err = fn()
    return err // Return either the function's error or the recovered panic error
}

// JumpReturn triggers a non-local return when there's an error.
// It converts the error to *ErrnieError before panicking.
func JumpReturn[T any](v T, err error) T {
	if err != nil {
		// Use the toErrnieError function to convert/wrap
		// Explicitly type ee as *ErrnieError
		var ee *ErrnieError = toErrnieError(err)
		if ee != nil {
			panic(Return{Value: v, Error: ee})
		}
	}
	return v
}

// Safe wraps a function call that might panic with errnie.Return.
// If such a panic occurs, it recovers and re-panics with just the *ErrnieError.
// This simplifies recovery logic higher up the stack, which only needs to recover *ErrnieError.
// If a different panic occurs, it's re-panicked as is.
func Safe[T any](fn func() T) (result T) {
	defer func() {
		if r := recover(); r != nil {
			if ret, ok := r.(Return); ok {
				// Re-panic with just the ErrnieError for simpler upstream recovery.
				panic(ret.Error)
			} else {
				// Re-panic if it's not our Return type
				panic(r)
			}
		}
	}()

	// Execute the function. If JumpReturn panics inside fn, the defer will catch it.
	result = fn()
	return // Return the result if no panic occurred
}

// Try remains a helper for functions returning (T, error).
// It uses Safe to wrap the call to JumpReturn.
func Try[T any](v T, err error) T {
	// Safe will catch the panic from JumpReturn (if err != nil)
	// and re-panic with just the *ErrnieError.
	return Safe(func() T {
		return JumpReturn(v, err)
	})
}

// toErrnieError converts a standard error to *ErrnieError.
// Renamed from Error to avoid conflict with the Error() method.
func toErrnieError(err error) *ErrnieError {
	if err == nil {
		return nil
	}
	if ee, ok := err.(*ErrnieError); ok {
		return ee // Already an ErrnieError
	}
	// Default conversion using InternalError
	return InternalError(err, err.Error())
}
