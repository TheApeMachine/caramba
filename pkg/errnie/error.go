/*
Package errnie provides a robust error handling system that integrates with Go's standard error
handling and HTTP status codes. It supports error wrapping, stack traces, and rich error context.
*/
package errnie

import (
	"fmt"
	"math"
	"net/http"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/spf13/viper"
)

var (
	sentryEnabled   bool
	errorAggregator = &ErrorAggregator{
		window:    time.Minute,
		threshold: 100,
		errors:    make(map[string]*aggregatedError),
	}
)

type aggregatedError struct{}

type ErrorAggregator struct {
	mu        sync.RWMutex
	window    time.Duration
	threshold int
	errors    map[string]*aggregatedError
}

// Reset clears the state of the error aggregator.
func (ea *ErrorAggregator) Reset() {
	ea.mu.Lock()
	defer ea.mu.Unlock()
	ea.errors = make(map[string]*aggregatedError)
	// Optionally reset window and threshold to defaults if needed
	// ea.window = defaultWindow
	// ea.threshold = defaultThreshold
}

func init() {
	v := viper.GetViper()

	// Set default values
	v.SetDefault("settings.sentry.enabled", false)
	v.SetDefault("settings.sentry.dsn", "")
	v.SetDefault("settings.sentry.environment", "development")
	v.SetDefault("settings.sentry.release", "")
	v.SetDefault("settings.sentry.sample_rate", 1.0)

	sentryEnabled = v.GetBool("settings.sentry.enabled")

	if sentryEnabled {
		err := sentry.Init(sentry.ClientOptions{
			Dsn:         v.GetString("settings.sentry.dsn"),
			Environment: v.GetString("settings.sentry.environment"),
			Release:     v.GetString("settings.sentry.release"),
			SampleRate:  v.GetFloat64("settings.sentry.sample_rate"),
		})
		if err != nil {
			// If Sentry initialization fails, disable it
			sentryEnabled = false
			fmt.Printf("Failed to initialize sentry: %v\n", err)
		}
	}
}

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

type RetryPolicy struct {
	MaxAttempts int
	Delay       time.Duration
	BackoffFunc func(attempt int, delay time.Duration) time.Duration
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
	errorType   ErrnieErrorType
	errors      []error
	messages    []string
	status      ErrnieStatusType
	stack       string
	level       sentry.Level
	context     map[string]interface{}
	breadcrumbs []sentry.Breadcrumb
	transaction string
	fingerprint []string
	retryPolicy *RetryPolicy
	attempts    int
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
		errorType:   UnknownError,
		errors:      make([]error, 0),
		messages:    make([]string, 0),
		status:      InternalServerErrorStatus,
		level:       sentry.LevelError,
		context:     make(map[string]interface{}),
		breadcrumbs: make([]sentry.Breadcrumb, 0),
		fingerprint: make([]string, 0),
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

	// Report to Sentry if enabled and should report based on aggregation
	if sentryEnabled {
		sentry.WithScope(func(scope *sentry.Scope) {
			scope.SetTag("error_type", err.errorType.String())
			scope.SetTag("status", fmt.Sprintf("%d", err.Status()))
			scope.SetLevel(err.level)
			scope.SetContext("error_context", err.context)
			scope.SetExtra("stack", err.stack)

			if err.transaction != "" {
				scope.SetTag("transaction", err.transaction)
			}

			if len(err.fingerprint) > 0 {
				scope.SetFingerprint(err.fingerprint)
			}

			for _, breadcrumb := range err.breadcrumbs {
				scope.AddBreadcrumb(&breadcrumb, 0)
			}

			sentry.CaptureException(err)
		})
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
func WithMessage(message ...string) ErrnieErrorOption {
	return func(e *ErrnieError) {
		for _, msg := range message {
			if msg != "" {
				e.messages = append(e.messages, msg)
			}
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

/*
WithLevel sets the error level
*/
func WithLevel(level sentry.Level) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.level = level
	}
}

/*
WithContext adds context information to the error
*/
func WithContext(key string, value interface{}) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.context[key] = value
	}
}

/*
WithBreadcrumb adds a breadcrumb to track error history
*/
func WithBreadcrumb(category, message string, data map[string]interface{}) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.breadcrumbs = append(e.breadcrumbs, sentry.Breadcrumb{
			Category: category,
			Message:  message,
			Data:     data,
			Level:    e.level,
			Type:     "error",
		})
	}
}

/*
WithTransaction sets the transaction name for better error grouping
*/
func WithTransaction(name string) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.transaction = name
	}
}

/*
WithFingerprint sets custom fingerprint rules for error grouping
*/
func WithFingerprint(fingerprint ...string) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.fingerprint = fingerprint
	}
}

/*
TraceError creates a new error with tracing context from a parent error
*/
func TraceError(parent *ErrnieError, options ...ErrnieErrorOption) *ErrnieError {
	if parent == nil {
		return New(options...)
	}

	// Inherit context and breadcrumbs from parent
	newOptions := append([]ErrnieErrorOption{
		WithLevel(parent.level),
		WithTransaction(parent.transaction),
	}, options...)

	// Add parent's context
	for k, v := range parent.context {
		newOptions = append(newOptions, WithContext(k, v))
	}

	// Add parent's breadcrumbs
	for _, b := range parent.breadcrumbs {
		newOptions = append(newOptions, WithBreadcrumb(b.Category, b.Message, b.Data))
	}

	return New(newOptions...)
}

// DefaultBackoff implements exponential backoff
func DefaultBackoff(attempt int, delay time.Duration) time.Duration {
	return delay * time.Duration(math.Pow(2, float64(attempt)))
}

// WithRetryPolicy sets a retry policy for recoverable errors
func WithRetryPolicy(maxAttempts int, delay time.Duration, backoff func(int, time.Duration) time.Duration) ErrnieErrorOption {
	if backoff == nil {
		backoff = DefaultBackoff
	}
	return func(e *ErrnieError) {
		e.retryPolicy = &RetryPolicy{
			MaxAttempts: maxAttempts,
			Delay:       delay,
			BackoffFunc: backoff,
		}
	}
}

// Retry attempts to recover from the error using the retry policy
func (e *ErrnieError) Retry(operation func() error) error {
	if e.retryPolicy == nil {
		return e
	}

	var lastErr error = e
	delay := e.retryPolicy.Delay

	for attempt := 0; attempt < e.retryPolicy.MaxAttempts; attempt++ {
		e.attempts++

		// Add retry attempt to breadcrumbs
		e.breadcrumbs = append(e.breadcrumbs, sentry.Breadcrumb{
			Category: "retry",
			Message:  fmt.Sprintf("Retry attempt %d/%d", e.attempts, e.retryPolicy.MaxAttempts),
			Level:    e.level,
			Data: map[string]interface{}{
				"delay": delay.String(),
			},
		})

		// Wait before retrying
		time.Sleep(delay)

		// Attempt the operation
		if err := operation(); err == nil {
			return nil
		} else {
			lastErr = err
			delay = e.retryPolicy.BackoffFunc(attempt, e.retryPolicy.Delay)
		}
	}

	return Wrap(lastErr, "all retry attempts failed")
}

// IsRetryable returns true if the error has a retry policy and hasn't exceeded max attempts
func (e *ErrnieError) IsRetryable() bool {
	return e.retryPolicy != nil && e.attempts < e.retryPolicy.MaxAttempts
}

// Example helper for common retry scenarios
func WithDefaultRetryPolicy() ErrnieErrorOption {
	return WithRetryPolicy(3, time.Second, DefaultBackoff)
}

// WithErrorAggregation configures error aggregation settings
func WithErrorAggregation(window time.Duration, threshold int) ErrnieErrorOption {
	return func(e *ErrnieError) {
		errorAggregator.window = window
		errorAggregator.threshold = threshold
	}
}
