package errnie

import (
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

const (
	multipleMessage        = "When creating an error with multiple messages"
	notFound               = "When creating an error with nil underlying error"
	badRequest             = "When creating an error with an underlying error"
	invalidInput           = "When creating an error with invalid input"
	validationInvalidInput = "[VALIDATION] invalid input"
	testError              = "test error"
	originalError          = "original error"
	secondError            = "second error"
	otherError             = "other error"
	fieldRequired          = "field required"
	wrappedError           = "wrapped error"
	resourceNotFound       = "resource not found"
	testMessage            = "test message"
	invalidRequest         = "invalid request"
	internalServerError    = "internal server error"
	unauthorizedAccess     = "unauthorized access"
	accessForbidden        = "access forbidden"
	requestTimeout         = "request timeout"
	wrappedMessage         = "wrapped message"
	withOtherOptions       = "When using with other options"
	initialError           = "initial error"
)

func TestRun(t *testing.T) {
	Convey("Given various error types", t, func() {
		testCases := []struct {
			name    string
			errType ErrnieErrorType
			want    string
		}{
			{"NilError", NilError, "NIL"},
			{"UnknownError", UnknownError, "UNKNOWN"},
			{"IOError", IOError, "IO"},
			{"ValidationError", ValidationError, "VALIDATION"},
			{"NetworkError", NetworkError, "NETWORK"},
			{"DatabaseError", DatabaseError, "DATABASE"},
			{"AuthenticationError", AuthenticationError, "AUTHENTICATION"},
			{"AuthorizationError", AuthorizationError, "AUTHORIZATION"},
			{"ConfigurationError", ConfigurationError, "CONFIGURATION"},
			{"ResourceNotFoundError", ResourceNotFoundError, "RESOURCE_NOT_FOUND"},
			{"ResourceConflictError", ResourceConflictError, "RESOURCE_CONFLICT"},
			{"TimeoutError", TimeoutError, "TIMEOUT"},
			{"RateLimitError", RateLimitError, "RATE_LIMIT"},
			{"InvalidInputError", InvalidInputError, "INVALID_INPUT"},
			{"SystemError", SystemError, "SYSTEM"},
			{"DependencyError", DependencyError, "DEPENDENCY"},
			{"UndefinedError", ErrnieErrorType(999), "UNKNOWN"},
		}

		for _, tc := range testCases {
			Convey(fmt.Sprintf("When converting %s (%d) to string", tc.name, tc.errType), func() {
				got := tc.errType.String()
				Convey(fmt.Sprintf("Then it should return '%s'", tc.want), func() {
					So(got, ShouldEqual, tc.want)
				})
			})
		}
	})
}

func TestError(t *testing.T) {
	Convey("Given various error scenarios", t, func() {
		Convey("When creating an error with just a type", func() {
			err := New(
				WithType(ValidationError),
				WithError(errors.New(invalidInput)),
			)
			So(err.Error(), ShouldEqual, validationInvalidInput)
		})

		Convey("When creating an error with a message", func() {
			err := New(WithType(ValidationError), WithMessage(invalidInput))
			So(err.Error(), ShouldEqual, validationInvalidInput)
		})

		Convey(multipleMessage, func() {
			err := New(
				WithType(ValidationError),
				WithMessage(invalidInput),
				WithMessage(fieldRequired),
			)
			So(err.Error(), ShouldEqual, "[VALIDATION] invalid input field required")
		})

		Convey("When creating an error with wrapped errors", func() {
			originalErr := errors.New(originalError)
			wrappedErr := New(
				WithType(ValidationError),
				WithError(originalErr),
				WithMessage(wrappedError),
			)
			So(wrappedErr.Error(), ShouldEqual, "[VALIDATION] wrapped error: original error")
		})
	})
}

func TestStatus(t *testing.T) {
	Convey("Given various status types", t, func() {
		testCases := []struct {
			status ErrnieStatusType
			want   int
		}{
			{OKStatus, http.StatusOK},
			{BadRequestStatus, http.StatusBadRequest},
			{UnauthorizedStatus, http.StatusUnauthorized},
			{ForbiddenStatus, http.StatusForbidden},
			{NotFoundStatus, http.StatusNotFound},
			{ConflictStatus, http.StatusConflict},
			{TooManyRequestsStatus, http.StatusTooManyRequests},
			{ServiceUnavailableStatus, http.StatusServiceUnavailable},
			{GatewayTimeoutStatus, http.StatusGatewayTimeout},
			{ErrnieStatusType(999), http.StatusInternalServerError},
		}

		for _, tc := range testCases {
			Convey(fmt.Sprintf("When getting HTTP status for %v", tc.status), func() {
				err := New(WithStatus(tc.status))
				got := err.Status()
				Convey("Then it should return the correct HTTP status code", func() {
					So(got, ShouldEqual, tc.want)
				})
			})
		}
	})
}

func TestType(t *testing.T) {
	Convey("Given an error with a specific type", t, func() {
		err := New(WithType(ValidationError))
		Convey("When getting the error type", func() {
			got := err.Type()
			Convey("Then it should return the correct type", func() {
				So(got, ShouldEqual, ValidationError)
			})
		})
	})
}

func TestStack(t *testing.T) {
	Convey("Given a new error", t, func() {
		err := New(WithMessage(testError))
		Convey("When getting the stack trace", func() {
			stack := err.Stack()
			Convey("Then it should contain stack information", func() {
				So(stack, ShouldNotBeEmpty)
				So(stack, ShouldContainSubstring, "error_test.go")
			})
		})
	})
}

func TestUnwrap(t *testing.T) {
	Convey("Given various wrapped error scenarios", t, func() {
		Convey("When unwrapping an error with no wrapped errors", func() {
			err := New(WithMessage(testError))
			unwrapped := err.Unwrap()
			So(unwrapped, ShouldBeNil)
		})

		Convey("When unwrapping an error with a wrapped error", func() {
			originalErr := errors.New(originalError)
			err := New(WithError(originalErr))
			unwrapped := err.Unwrap()
			So(unwrapped, ShouldEqual, originalErr)
		})

		Convey("When unwrapping an error with multiple wrapped errors", func() {
			originalErr := errors.New(originalError)
			secondErr := errors.New(secondError)
			err := New(WithError(originalErr), WithError(secondErr))
			unwrapped := err.Unwrap()
			So(unwrapped, ShouldEqual, originalErr)
		})
	})
}

func TestIs(t *testing.T) {
	Convey("Given various error comparison scenarios", t, func() {
		originalErr := errors.New(originalError)

		Convey("When comparing with a contained error", func() {
			err := New(WithError(originalErr))
			So(err.Is(originalErr), ShouldBeTrue)
		})

		Convey("When comparing with a different error", func() {
			err := New(WithError(originalErr))
			otherErr := errors.New(otherError)
			So(err.Is(otherErr), ShouldBeFalse)
		})

		Convey("When comparing with nil", func() {
			err := New(WithError(originalErr))
			So(err.Is(nil), ShouldBeFalse)
		})
	})
}

func TestNew(t *testing.T) {
	Convey("Given various error creation scenarios", t, func() {
		Convey("When creating a new error with no options", func() {
			err := New()
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, UnknownError)
			So(err.Status(), ShouldEqual, http.StatusInternalServerError)
		})

		Convey("When creating a new error with all options", func() {
			originalErr := errors.New(originalError)
			err := New(
				WithType(ValidationError),
				WithStatus(BadRequestStatus),
				WithError(originalErr),
				WithMessage(testMessage),
			)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, ValidationError)
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
			So(err.Error(), ShouldContainSubstring, testMessage)
			So(err.Error(), ShouldContainSubstring, originalError)
		})
	})
}

func TestNotFound(t *testing.T) {
	Convey("Given the NotFound helper function", t, func() {
		originalErr := errors.New(originalError)

		Convey(notFound, func() {
			err := NotFound(nil, resourceNotFound)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, ResourceNotFoundError)
			So(err.Status(), ShouldEqual, http.StatusNotFound)
			So(err.Error(), ShouldContainSubstring, resourceNotFound)
		})

		Convey(badRequest, func() {
			err := NotFound(originalErr, resourceNotFound)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, ResourceNotFoundError)
			So(err.Status(), ShouldEqual, http.StatusNotFound)
			So(err.Error(), ShouldContainSubstring, resourceNotFound)
			So(err.Error(), ShouldContainSubstring, originalError)
		})

		Convey("multipleMessage", func() {
			err := NotFound(originalErr, "resource", "not", "found")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "resource not found")
		})
	})
}

func TestBadRequest(t *testing.T) {
	Convey("Given the BadRequest helper function", t, func() {
		originalErr := errors.New(originalError)

		Convey(notFound, func() {
			err := BadRequest(nil, invalidRequest)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, InvalidInputError)
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
			So(err.Error(), ShouldContainSubstring, invalidRequest)
		})

		Convey(badRequest, func() {
			err := BadRequest(originalErr, invalidRequest)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, InvalidInputError)
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
			So(err.Error(), ShouldContainSubstring, invalidRequest)
			So(err.Error(), ShouldContainSubstring, originalError)
		})

		Convey("multipleMessage", func() {
			err := BadRequest(originalErr, "invalid", "request", "format")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "invalid request format")
		})
	})
}

func TestInternalError(t *testing.T) {
	Convey("Given the InternalError helper function", t, func() {
		originalErr := errors.New(originalError)

		Convey(notFound, func() {
			err := InternalError(nil, internalServerError)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, SystemError)
			So(err.Status(), ShouldEqual, http.StatusInternalServerError)
			So(err.Error(), ShouldContainSubstring, internalServerError)
		})

		Convey(badRequest, func() {
			err := InternalError(originalErr, internalServerError)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, SystemError)
			So(err.Status(), ShouldEqual, http.StatusInternalServerError)
			So(err.Error(), ShouldContainSubstring, internalServerError)
			So(err.Error(), ShouldContainSubstring, originalError)
		})

		Convey("multipleMessage", func() {
			err := InternalError(originalErr, "internal", "processing", "failed")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "internal processing failed")
		})
	})
}

func TestUnauthorized(t *testing.T) {
	Convey("Given the Unauthorized helper function", t, func() {
		originalErr := errors.New(originalError)

		Convey(notFound, func() {
			err := Unauthorized(nil, unauthorizedAccess)
			So(err, ShouldBeNil)
		})

		Convey(badRequest, func() {
			err := Unauthorized(originalErr, unauthorizedAccess)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, AuthenticationError)
			So(err.Status(), ShouldEqual, http.StatusUnauthorized)
			So(err.Error(), ShouldContainSubstring, unauthorizedAccess)
			So(err.Error(), ShouldContainSubstring, originalError)
		})

		Convey("multipleMessage", func() {
			err := Unauthorized(originalErr, "invalid", "credentials", "provided")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "invalid credentials provided")
		})
	})
}

func TestForbidden(t *testing.T) {
	Convey("Given the Forbidden helper function", t, func() {
		originalErr := errors.New(originalError)

		Convey(notFound, func() {
			err := Forbidden(nil, accessForbidden)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, AuthorizationError)
			So(err.Status(), ShouldEqual, http.StatusForbidden)
			So(err.Error(), ShouldContainSubstring, accessForbidden)
		})

		Convey(badRequest, func() {
			err := Forbidden(originalErr, accessForbidden)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, AuthorizationError)
			So(err.Status(), ShouldEqual, http.StatusForbidden)
			So(err.Error(), ShouldContainSubstring, accessForbidden)
			So(err.Error(), ShouldContainSubstring, originalError)
		})

		Convey("multipleMessage", func() {
			err := Forbidden(originalErr, "insufficient", "permissions", "for", "resource")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "insufficient permissions for resource")
		})
	})
}

func TestTimeout(t *testing.T) {
	Convey("Given the Timeout helper function", t, func() {
		originalErr := errors.New(originalError)

		Convey(notFound, func() {
			err := Timeout(nil, requestTimeout)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, TimeoutError)
			So(err.Status(), ShouldEqual, http.StatusGatewayTimeout)
			So(err.Error(), ShouldContainSubstring, requestTimeout)
		})

		Convey(badRequest, func() {
			err := Timeout(originalErr, requestTimeout)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, TimeoutError)
			So(err.Status(), ShouldEqual, http.StatusGatewayTimeout)
			So(err.Error(), ShouldContainSubstring, requestTimeout)
			So(err.Error(), ShouldContainSubstring, originalError)
		})

		Convey("multipleMessage", func() {
			err := Timeout(originalErr, "operation", "timed", "out")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "operation timed out")
		})
	})
}

func TestWrap(t *testing.T) {
	Convey("Given various error wrapping scenarios", t, func() {
		Convey("When wrapping nil error", func() {
			err := Wrap(nil, wrappedMessage)
			So(err, ShouldBeNil)
		})

		Convey("When wrapping an error with a message", func() {
			originalErr := errors.New(originalError)
			err := Wrap(originalErr, wrappedMessage)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, wrappedMessage)
			So(err.Error(), ShouldContainSubstring, originalError)
		})

		Convey("When wrapping an error with a formatted message", func() {
			originalErr := errors.New(originalError)
			err := Wrap(originalErr, "wrapped message: %v", "detail")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "wrapped message: detail")
			So(err.Error(), ShouldContainSubstring, originalError)
		})
	})
}

func TestWithType(t *testing.T) {
	Convey("Given the WithType option function", t, func() {
		Convey("When setting a specific error type", func() {
			err := New(WithType(ValidationError))
			So(err.Type(), ShouldEqual, ValidationError)
		})

		Convey("When setting multiple error types", func() {
			err := New(
				WithType(ValidationError),
				WithType(DatabaseError),
			)
			So(err.Type(), ShouldEqual, DatabaseError)
		})

		Convey(withOtherOptions, func() {
			err := New(
				WithType(ValidationError),
				WithMessage(testMessage),
			)
			So(err.Type(), ShouldEqual, ValidationError)
			So(err.Error(), ShouldContainSubstring, testMessage)
		})
	})
}

func TestWithError(t *testing.T) {
	Convey("Given the WithError option function", t, func() {
		originalErr := errors.New(originalError)

		Convey("When adding a nil error", func() {
			err := New(WithError(nil))
			So(len(err.errors), ShouldEqual, 0)
		})

		Convey("When adding a valid error", func() {
			err := New(WithError(originalErr))
			So(len(err.errors), ShouldEqual, 1)
			So(err.errors[0], ShouldEqual, originalErr)
			So(err.Error(), ShouldContainSubstring, originalError)
		})

		Convey("When adding multiple errors", func() {
			secondErr := errors.New(secondError)
			err := New(
				WithError(originalErr),
				WithError(secondErr),
			)
			So(len(err.errors), ShouldEqual, 2)
			So(err.errors[0], ShouldEqual, originalErr)
			So(err.errors[1], ShouldEqual, secondErr)
			So(err.Error(), ShouldContainSubstring, originalError)
			So(err.Error(), ShouldContainSubstring, secondError)
		})

		Convey(withOtherOptions, func() {
			err := New(
				WithError(originalErr),
				WithMessage(testMessage),
			)
			So(len(err.errors), ShouldEqual, 1)
			So(err.Error(), ShouldContainSubstring, testMessage)
			So(err.Error(), ShouldContainSubstring, originalError)
		})
	})
}

func TestWithMessage(t *testing.T) {
	Convey("Given the WithMessage option function", t, func() {
		Convey("When adding an empty message", func() {
			err := New(WithMessage(""))
			So(len(err.messages), ShouldEqual, 0)
			So(err.Error(), ShouldEqual, "[UNKNOWN] error")
		})

		Convey("When adding a valid message", func() {
			err := New(WithMessage(testMessage))
			So(len(err.messages), ShouldEqual, 1)
			So(err.messages[0], ShouldEqual, testMessage)
			So(err.Error(), ShouldContainSubstring, testMessage)
		})

		Convey("When adding multiple messages", func() {
			err := New(
				WithMessage(testMessage),
				WithMessage(secondError),
			)
			So(len(err.messages), ShouldEqual, 2)
			So(err.messages[0], ShouldEqual, testMessage)
			So(err.messages[1], ShouldEqual, secondError)
			So(err.Error(), ShouldContainSubstring, testMessage)
			So(err.Error(), ShouldContainSubstring, secondError)
		})

		Convey(withOtherOptions, func() {
			originalErr := errors.New(originalError)
			err := New(
				WithMessage(testMessage),
				WithError(originalErr),
			)
			So(len(err.messages), ShouldEqual, 1)
			So(err.Error(), ShouldContainSubstring, testMessage)
			So(err.Error(), ShouldContainSubstring, originalError)
		})
	})
}

func TestWithStatus(t *testing.T) {
	Convey("Given the WithStatus option function", t, func() {
		Convey("When setting a specific status", func() {
			err := New(WithStatus(BadRequestStatus))
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
		})

		Convey("When setting multiple statuses", func() {
			err := New(
				WithStatus(BadRequestStatus),
				WithStatus(NotFoundStatus),
			)
			So(err.Status(), ShouldEqual, http.StatusNotFound)
		})

		Convey("When using default status", func() {
			err := New()
			So(err.Status(), ShouldEqual, http.StatusInternalServerError)
		})

		Convey(withOtherOptions, func() {
			err := New(
				WithStatus(BadRequestStatus),
				WithMessage(testMessage),
				WithType(ValidationError),
			)
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
			So(err.Type(), ShouldEqual, ValidationError)
			So(err.Error(), ShouldContainSubstring, testMessage)
		})

		Convey("When using an undefined status", func() {
			err := New(WithStatus(ErrnieStatusType(999)))
			So(err.Status(), ShouldEqual, http.StatusInternalServerError)
		})
	})
}

func TestRetryPolicy(t *testing.T) {
	Convey("Given various retry scenarios", t, func() {
		Convey("When using default retry policy", func() {
			attempts := 0
			operation := func() error {
				attempts++
				if attempts < 2 {
					return errors.New("temporary error")
				}
				return nil
			}

			err := New(
				WithDefaultRetryPolicy(),
				WithMessage(initialError),
			)

			retryErr := err.Retry(operation)
			So(retryErr, ShouldBeNil)
			So(attempts, ShouldEqual, 2)
			So(err.attempts, ShouldEqual, 2)
		})

		Convey("When all retry attempts fail", func() {
			attempts := 0
			operation := func() error {
				attempts++
				return errors.New("persistent error")
			}

			err := New(
				WithRetryPolicy(2, time.Millisecond, nil),
				WithMessage(initialError),
			)

			retryErr := err.Retry(operation)
			So(retryErr, ShouldNotBeNil)
			So(retryErr.Error(), ShouldContainSubstring, "all retry attempts failed")
			So(attempts, ShouldEqual, 2)
			So(err.attempts, ShouldEqual, 2)
		})

		Convey("When using custom backoff", func() {
			customBackoff := func(attempt int, delay time.Duration) time.Duration {
				return delay * time.Duration(attempt+1)
			}

			err := New(
				WithRetryPolicy(3, time.Millisecond, customBackoff),
				WithMessage(initialError),
			)

			So(err.IsRetryable(), ShouldBeTrue)
			So(err.retryPolicy.BackoffFunc, ShouldNotBeNil)
			So(err.retryPolicy.MaxAttempts, ShouldEqual, 3)
			So(err.retryPolicy.Delay, ShouldEqual, time.Millisecond)
		})
	})
}
