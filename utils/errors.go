package utils

import (
	"encoding/json"
	"fmt"
	"time"
)

// ErrorSeverity represents the severity level of an error
type ErrorSeverity int

const (
	SeverityLow ErrorSeverity = iota
	SeverityMedium
	SeverityHigh
	SeverityCritical
)

// RetryStrategy defines how retries should be handled
type RetryStrategy interface {
	NextDelay(attempt int) time.Duration
	ShouldRetry(err error) bool
	MaxAttempts() int
}

// RetryInfo contains information about retry attempts
type RetryInfo struct {
	Attempts    int
	MaxAttempts int
	NextRetry   time.Time
	Strategy    RetryStrategy
}

// ErrorWithContext provides rich context around errors
type ErrorWithContext struct {
	Err       error
	Context   map[string]interface{}
	Timestamp time.Time
	Severity  ErrorSeverity
	RetryInfo *RetryInfo
}

// NewErrorWithContext creates a new ErrorWithContext
func NewErrorWithContext(err error, severity ErrorSeverity) *ErrorWithContext {
	return &ErrorWithContext{
		Err:       err,
		Context:   make(map[string]interface{}),
		Timestamp: time.Now(),
		Severity:  severity,
	}
}

// WithContext adds context to the error
func (e *ErrorWithContext) WithContext(key string, value interface{}) *ErrorWithContext {
	e.Context[key] = value
	return e
}

// WithRetry adds retry information to the error
func (e *ErrorWithContext) WithRetry(retryInfo *RetryInfo) *ErrorWithContext {
	e.RetryInfo = retryInfo
	return e
}

// Error implements the error interface
func (e *ErrorWithContext) Error() string {
	ctx, _ := json.Marshal(e.Context)
	return fmt.Sprintf("[%s] %v (Severity: %d, Context: %s)",
		e.Timestamp.Format(time.RFC3339),
		e.Err,
		e.Severity,
		string(ctx),
	)
}

// ExponentialBackoff implements RetryStrategy with exponential backoff
type ExponentialBackoff struct {
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Factor       float64
	maxAttempts  int
}

// NewExponentialBackoff creates a new ExponentialBackoff
func NewExponentialBackoff(initialDelay time.Duration, maxDelay time.Duration, factor float64, maxAttempts int) *ExponentialBackoff {
	return &ExponentialBackoff{
		InitialDelay: initialDelay,
		MaxDelay:     maxDelay,
		Factor:       factor,
		maxAttempts:  maxAttempts,
	}
}

// NextDelay calculates the next delay duration
func (eb *ExponentialBackoff) NextDelay(attempt int) time.Duration {
	delay := float64(eb.InitialDelay) * pow(eb.Factor, float64(attempt))
	if delay > float64(eb.MaxDelay) {
		return eb.MaxDelay
	}
	return time.Duration(delay)
}

// ShouldRetry determines if a retry should be attempted
func (eb *ExponentialBackoff) ShouldRetry(err error) bool {
	// Add logic here to determine if the error is retryable
	// For now, we'll assume all errors are retryable
	return true
}

// MaxAttempts returns the maximum number of retry attempts
func (eb *ExponentialBackoff) MaxAttempts() int {
	return eb.maxAttempts
}

// Helper function for power calculation
func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}
