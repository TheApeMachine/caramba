/*
Package errnie provides a robust error handling system that integrates with Go's standard error
handling and HTTP status codes. It supports error wrapping, stack traces, and rich error context.
*/
package errnie

import (
	"fmt"
	"net/http"
	"runtime"
	"slices"
	"strings"
)

/*
ErrnieErrorType represents the classification of an error.
It helps categorize errors for better error handling and logging.
*/
type ErrnieErrorType uint

/*
ErrnieStatusType represents an HTTP status code.
It maps internal error types to appropriate HTTP responses.
*/
type ErrnieStatusType uint

const (
	/*
		Error Types define the various categories of errors that can occur.
		These help in programmatic error handling and provide better context
		for debugging.

		Example:
			if err.Type() == ValidationError {
				// Handle validation error
			}
	*/
	NilError              ErrnieErrorType = iota // Represents no error
	UnknownError                                 // Error type cannot be determined
	IOError                                      // Input/Output related errors
	ValidationError                              // Data validation errors
	NetworkError                                 // Network-related errors
	DatabaseError                                // Database operation errors
	AuthenticationError                          // Authentication failures
	AuthorizationError                           // Authorization/permission errors
	ConfigurationError                           // Configuration-related errors
	ResourceNotFoundError                        // Resource not found errors
	ResourceConflictError                        // Resource conflict errors
	TimeoutError                                 // Timeout-related errors
	RateLimitError                               // Rate limiting errors
	InvalidInputError                            // Invalid input errors
	SystemError                                  // Internal system errors
	DependencyError                              // External dependency errors

	/*
		Status Types define the HTTP status codes that can be returned.
		These map internal error types to appropriate HTTP responses.

		Example:
			if err.Status() == http.StatusNotFound {
				// Handle 404 error
			}
	*/
	NilStatus                 ErrnieStatusType = iota // No status
	UnknownStatus                                     // Unknown status
	OKStatus                                          // 200 OK
	BadRequestStatus                                  // 400 Bad Request
	UnauthorizedStatus                                // 401 Unauthorized
	ForbiddenStatus                                   // 403 Forbidden
	NotFoundStatus                                    // 404 Not Found
	ConflictStatus                                    // 409 Conflict
	TooManyRequestsStatus                             // 429 Too Many Requests
	InternalServerErrorStatus                         // 500 Internal Server Error
	ServiceUnavailableStatus                          // 503 Service Unavailable
	GatewayTimeoutStatus                              // 504 Gateway Timeout
)

/*
String returns a human-readable representation of the error type.
This is useful for logging and error messages.

Example:

	errType := ValidationError
	fmt.Println(errType.String()) // Prints: "VALIDATION"
*/
func (e ErrnieErrorType) String() string {
	switch e {
	case NilError:
		return "NIL"
	case UnknownError:
		return "UNKNOWN"
	case IOError:
		return "IO"
	case ValidationError:
		return "VALIDATION"
	case NetworkError:
		return "NETWORK"
	case DatabaseError:
		return "DATABASE"
	case AuthenticationError:
		return "AUTHENTICATION"
	case AuthorizationError:
		return "AUTHORIZATION"
	case ConfigurationError:
		return "CONFIGURATION"
	case ResourceNotFoundError:
		return "RESOURCE_NOT_FOUND"
	case ResourceConflictError:
		return "RESOURCE_CONFLICT"
	case TimeoutError:
		return "TIMEOUT"
	case RateLimitError:
		return "RATE_LIMIT"
	case InvalidInputError:
		return "INVALID_INPUT"
	case SystemError:
		return "SYSTEM"
	case DependencyError:
		return "DEPENDENCY"
	default:
		return "UNKNOWN"
	}
}

/*
ErrnieError represents a rich error type that includes context about the error,
such as error type, messages, wrapped errors, HTTP status, and stack trace.

Example:

	err := &ErrnieError{
		errorType: ValidationError,
		messages: []string{"invalid input"},
		status: BadRequestStatus,
	}
*/
type ErrnieError struct {
	errorType ErrnieErrorType
	errors    []error
	messages  []string
	status    ErrnieStatusType
	stack     string
}

/*
Error implements the error interface and returns a formatted error message.
The message includes the error type and all associated messages and wrapped errors.

Example:

	err := New(WithType(ValidationError), WithMessage("invalid input"))
	fmt.Println(err.Error()) // Prints: "[VALIDATION] invalid input"
*/
func (e *ErrnieError) Error() string {
	var parts []string

	if len(e.messages) > 0 {
		messages := make([]string, 0, len(e.messages))
		for _, msg := range e.messages {
			if msg != "" {
				messages = append(messages, msg)
			}
		}
		if len(messages) > 0 {
			parts = append(parts, strings.Join(messages, " "))
		}
	}

	if len(e.errors) > 0 {
		for _, e := range e.errors {
			parts = append(parts, e.Error())
		}
	}

	if len(parts) == 0 {
		parts = append(parts, "error")
	}

	return fmt.Sprintf("[%s] %s", e.errorType, strings.Join(parts, ": "))
}

/*
Status returns the HTTP status code associated with the error.
This is useful for HTTP handlers to return appropriate status codes.

Example:

	err := NotFound(nil, "user not found")
	status := err.Status() // Returns http.StatusNotFound
*/
func (e *ErrnieError) Status() int {
	switch e.status {
	case OKStatus:
		return http.StatusOK
	case BadRequestStatus:
		return http.StatusBadRequest
	case UnauthorizedStatus:
		return http.StatusUnauthorized
	case ForbiddenStatus:
		return http.StatusForbidden
	case NotFoundStatus:
		return http.StatusNotFound
	case ConflictStatus:
		return http.StatusConflict
	case TooManyRequestsStatus:
		return http.StatusTooManyRequests
	case ServiceUnavailableStatus:
		return http.StatusServiceUnavailable
	case GatewayTimeoutStatus:
		return http.StatusGatewayTimeout
	default:
		return http.StatusInternalServerError
	}
}

/*
Type returns the error type of the error.
This is useful for programmatic error handling.

Example:

	err := New(WithType(ValidationError))
	if err.Type() == ValidationError {
		// Handle validation error
	}
*/
func (e *ErrnieError) Type() ErrnieErrorType {
	return e.errorType
}

/*
Stack returns the stack trace captured when the error was created.
This is useful for debugging and error tracking.

Example:

	err := New(WithMessage("something went wrong"))
	fmt.Println(err.Stack()) // Prints the stack trace
*/
func (e *ErrnieError) Stack() string {
	return e.stack
}

/*
Unwrap implements the errors.Unwrap interface and returns the first wrapped error.
This integrates with Go's error wrapping system.

Example:

	originalErr := errors.New("original error")
	err := New(WithError(originalErr))
	unwrapped := errors.Unwrap(err) // Returns originalErr
*/
func (e *ErrnieError) Unwrap() error {
	if len(e.errors) > 0 {
		return e.errors[0]
	}
	return nil
}

/*
Is implements the errors.Is interface for error comparison.
This integrates with Go's error comparison system.

Example:

	originalErr := errors.New("original error")
	err := New(WithError(originalErr))
	if errors.Is(err, originalErr) {
		// Handle matching error
	}
*/
func (e *ErrnieError) Is(target error) bool {
	return slices.Contains(e.errors, target)
}

/*
ErrnieErrorOption is a function type for configuring ErrnieError instances.
It's used with the New function to create errors with specific configurations.
*/
type ErrnieErrorOption func(*ErrnieError)

/*
New creates a new ErrnieError with the provided options.
It captures a stack trace and initializes the error with default values.

Example:

	err := New(
		WithType(ValidationError),
		WithMessage("invalid input"),
		WithStatus(BadRequestStatus),
	)
*/
func New(options ...ErrnieErrorOption) *ErrnieError {
	err := &ErrnieError{
		errorType: UnknownError,
		errors:    make([]error, 0),
		messages:  make([]string, 0),
		status:    InternalServerErrorStatus,
	}

	// Capture stack trace
	buf := make([]byte, 4096)
	n := runtime.Stack(buf, false)
	err.stack = string(buf[:n])

	for _, option := range options {
		option(err)
	}

	if len(err.errors) > 0 {
		Error(err.errors[len(err.errors)-1])
	}

	return err
}

/*
NotFound creates a new error for resource not found scenarios.
It sets appropriate error type and HTTP status code.

Example:

	err := NotFound(nil, "user with ID 123 not found")
*/
func NotFound(err error, msg ...string) *ErrnieError {
	return New(
		WithType(ResourceNotFoundError),
		WithStatus(NotFoundStatus),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

/*
BadRequest creates a new error for invalid request scenarios.
It sets appropriate error type and HTTP status code.

Example:

	err := BadRequest(nil, "invalid user ID format")
*/
func BadRequest(err error, msg ...string) *ErrnieError {
	return New(
		WithType(InvalidInputError),
		WithStatus(BadRequestStatus),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

/*
InternalError creates a new error for internal system errors.
It sets appropriate error type and HTTP status code.

Example:

	err := InternalError(err, "failed to process request")
*/
func InternalError(err error, msg ...string) *ErrnieError {
	return New(
		WithType(SystemError),
		WithStatus(InternalServerErrorStatus),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

/*
Unauthorized creates a new error for authentication failures.
It sets appropriate error type and HTTP status code.

Example:

	err := Unauthorized(nil, "invalid credentials")
*/
func Unauthorized(err error, msg ...string) *ErrnieError {
	return New(
		WithType(AuthenticationError),
		WithStatus(UnauthorizedStatus),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

/*
Forbidden creates a new error for authorization failures.
It sets appropriate error type and HTTP status code.

Example:

	err := Forbidden(nil, "insufficient permissions")
*/
func Forbidden(err error, msg ...string) *ErrnieError {
	return New(
		WithType(AuthorizationError),
		WithStatus(ForbiddenStatus),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

/*
Timeout creates a new error for timeout scenarios.
It sets appropriate error type and HTTP status code.

Example:

	err := Timeout(nil, "request timed out after 30s")
*/
func Timeout(err error, msg ...string) *ErrnieError {
	return New(
		WithType(TimeoutError),
		WithStatus(GatewayTimeoutStatus),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

/*
WithType sets the error type for an ErrnieError.

Example:

	err := New(WithType(ValidationError))
*/
func WithType(errorType ErrnieErrorType) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.errorType = errorType
	}
}

/*
WithError adds an error to the ErrnieError's wrapped errors.
Nil errors are ignored.

Example:

	originalErr := errors.New("original error")
	err := New(WithError(originalErr))
*/
func WithError(err error) ErrnieErrorOption {
	return func(e *ErrnieError) {
		if err != nil {
			e.errors = append(e.errors, err)
		}
	}
}

/*
WithMessage adds a message to the ErrnieError's messages.
Empty messages are ignored.

Example:

	err := New(WithMessage("something went wrong"))
*/
func WithMessage(message string) ErrnieErrorOption {
	return func(e *ErrnieError) {
		if message != "" {
			e.messages = append(e.messages, message)
		}
	}
}

/*
WithStatus sets the HTTP status code for an ErrnieError.

Example:

	err := New(WithStatus(BadRequestStatus))
*/
func WithStatus(status ErrnieStatusType) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.status = status
	}
}

/*
Wrap creates a new error that wraps an existing error with additional context.
If the original error is nil, it returns nil.

Example:

	if err := doSomething(); err != nil {
		return Wrap(err, "failed to do something: %v", err)
	}
*/
func Wrap(err error, msg string, args ...interface{}) *ErrnieError {
	if err == nil {
		return nil
	}

	message := fmt.Sprintf(msg, args...)
	return New(
		WithError(err),
		WithMessage(message),
	)
}
