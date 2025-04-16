package errnie

/*
Return holds the return value and a potential ErrnieError from a function
that might exit via a panic triggered by JumpReturn. It's used internally
by the Safe function to capture the results of a potentially panicking call.
*/
type Return struct {
	Value any
	Error *ErrnieError // Changed from error
}

// In pkg/errnie/safe.go (or a new file)

/*
RunSafely executes the given function `fn` and recovers from any panic
specifically caused by an *ErrnieError (typically initiated by functions like
errnie.Try or errnie.JumpReturn within `fn`). It allows centralizing panic
recovery logic for expected error conditions handled by Errnie.

Motivation: To provide a boundary where Errnie-based panics are caught and
converted back into standard errors, preventing them from crashing the program
while allowing other unexpected panics to propagate.

Usage:

	err := errnie.RunSafely(func() {
		// Code that might use errnie.Try or errnie.JumpReturn
		val := errnie.Try(potentiallyFailingOp())
		fmt.Println("Operation succeeded:", val)
	})
	if err != nil {
		fmt.Println("Operation failed:", err)
	}

It returns the recovered *ErrnieError as a standard error, or nil if no
ErrnieError panic occurred. If a different panic occurs, it is re-panicked.
*/
func RunSafely(fn func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			// Check if the panic was caused by an *ErrnieError
			if ee, ok := r.(*ErrnieError); ok {
				// Return the ErrnieError directly
				err = ee
			} else {
				// It was a different panic, re-panic
				panic(r)
			}
		}
	}()

	// Execute the user's function
	fn()
	return
}

/*
RunSafelyErr is similar to RunSafely but wraps a function `fn` that itself
returns an error. It recovers from *ErrnieError panics within `fn` and also
captures the error returned normally by `fn`.

Motivation: To handle both panic-based errors (via Errnie) and standard
return-based errors from the wrapped function within a single recovery block.

Usage:

	err := errnie.RunSafelyErr(func() error {
		// Code that might use errnie.Try or return an error
		val := errnie.Try(potentiallyFailingOp1())

		fmt.Println("Op1 succeeded:", val)

		err := potentiallyFailingOp2()

		if err != nil {
			return fmt.Errorf("op2 failed: %w", err) // Standard error return
		}

		return nil
	})
	if err != nil {
		fmt.Println("Operation failed:", err) // Catches ErrnieError panics OR returned errors
	}

It returns the recovered *ErrnieError or the error returned by `fn`. If a
different panic occurs, it is re-panicked.
*/
func RunSafelyErr(fn func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if ee, ok := r.(*ErrnieError); ok {
				err = ee
			} else {
				panic(r)
			}
		}
	}()

	// Execute the user's function and capture its return error
	if e := fn(); e != nil {
		err = toErrnieError(e)
	}
	return
}

/*
JumpReturn checks if the provided error `err` is non-nil. If it is, it converts
`err` into an *ErrnieError (if it isn't one already) and then panics with a
`Return` struct containing the original value `v` and the *ErrnieError.
This panic is intended to be caught by `Safe` or a `RunSafely` variant.

Motivation: To provide the core mechanism for Errnie's non-local return flow.
It bundles the original value and the error into a panic that can be intercepted.
It ensures that standard errors are wrapped into *ErrnieError for consistent handling.

Usage: Typically used indirectly via `Try`, but can be used directly:

	func processItem(id int) (ResultType, error) {
		// ... some processing ...
		data, err := fetchResource(id)
		// If fetchResource fails, panic with Return{Value: defaultResult, Error: *ErrnieError}
		checkedData := errnie.JumpReturn(data, err)
		// ... continue processing with checkedData ...
		return finalResult, nil
	}

If `err` is nil, it simply returns the value `v`.
*/
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

/*
Safe wraps a function call (`fn`) that might use `JumpReturn` (often indirectly via `Try`).
If `fn` panics with the specific `Return` struct used by `JumpReturn`, `Safe` recovers
from this panic, extracts the *ErrnieError from the `Return` struct, and then re-panics
with *only* the *ErrnieError.

Motivation: Simplifies the recovery logic further up the call stack (e.g., in `RunSafely`).
Instead of needing to recover the `Return` struct and check its type, upstream recovery
logic only needs to recover and check for *ErrnieError directly. It acts as an intermediary
panic handler.

Usage: Primarily used by `Try`, but could be used to wrap blocks:

	func complexOperation() string {
		// ...
		intermediateResult := errnie.Safe(func() SomeType {
			// This block might use JumpReturn or Try
			step1Result := errnie.Try(step1())
			step2Result := errnie.Try(step2(step1Result))
			return step2Result
		})
		// ... use intermediateResult ...
		return finalResult
	}

If a panic occurs within `fn` that is *not* the `Return` struct, `Safe` re-panics
that original panic value. If `fn` completes without a panic, `Safe` returns its result.
*/
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

/*
Try is a convenience function acting as syntactic sugar for checking a standard
Go multi-value return `(T, error)`. If the error is non-nil, it triggers the
Errnie panic-based error handling flow; otherwise, it returns the value `T`.

Motivation: Reduces the boilerplate of `if err != nil { ... }` checks common in Go,
allowing for a more linear flow of code when dealing with operations that return
`(value, error)`, especially within functions managed by `RunSafely` or `RunSafelyErr`.

Usage: Replaces standard error checking for functions returning (T, error).

	// Standard Go
	conn, err := db.Connect("connection_string")
	if err != nil {
		return fmt.Errorf("db connection failed: %w", err)
	}
	defer conn.Close()

	// Using errnie.Try (within a RunSafely block)
	conn := errnie.Try(db.Connect("connection_string"))
	defer conn.Close() // Executes only if Try doesn't panic

It uses `Safe` and `JumpReturn` internally. The panic triggered by `Try` (if `err`
is non-nil) contains only the *ErrnieError and is expected to be caught by a
`RunSafely` or `RunSafelyErr` block higher up the stack.
*/
func Try[T any](v T, err error) T {
	// Safe will catch the panic from JumpReturn (if err != nil)
	// and re-panic with just the *ErrnieError.
	return Safe(func() T {
		return JumpReturn(v, err)
	})
}

/*
toErrnieError converts or wraps a standard `error` into an `*ErrnieError`.

If the input `err` is nil, it returns nil.
If the input `err` is already an `*ErrnieError`, it returns it directly.
Otherwise, it wraps the standard `error` within a new `*ErrnieError`, using
`SystemError` as the default wrapper type.

Motivation: Ensures that all errors handled by the Errnie panic flow are
consistently of type *ErrnieError, simplifying type assertions in recovery blocks.
*/
func toErrnieError(err error) *ErrnieError {
	if err == nil {
		return nil
	}

	if ee, ok := err.(*ErrnieError); ok {
		return ee
	}

	// Preserve the original error type
	return New(
		WithType(SystemError),
		WithError(err),
		WithMessage(err.Error()),
	)
}
