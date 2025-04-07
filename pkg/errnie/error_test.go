package errnie

import (
	"errors"
	"fmt"
	"net/http"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestString(t *testing.T) {
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
			err := New(WithType(ValidationError))
			So(err.Error(), ShouldEqual, "[VALIDATION] error")
		})

		Convey("When creating an error with a message", func() {
			err := New(WithType(ValidationError), WithMessage("invalid input"))
			So(err.Error(), ShouldEqual, "[VALIDATION] invalid input")
		})

		Convey("When creating an error with multiple messages", func() {
			err := New(
				WithType(ValidationError),
				WithMessage("invalid input"),
				WithMessage("field required"),
			)
			So(err.Error(), ShouldEqual, "[VALIDATION] invalid input field required")
		})

		Convey("When creating an error with wrapped errors", func() {
			originalErr := errors.New("original error")
			wrappedErr := New(
				WithType(ValidationError),
				WithError(originalErr),
				WithMessage("wrapped error"),
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
		err := New(WithMessage("test error"))
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
			err := New(WithMessage("test error"))
			unwrapped := err.Unwrap()
			So(unwrapped, ShouldBeNil)
		})

		Convey("When unwrapping an error with a wrapped error", func() {
			originalErr := errors.New("original error")
			err := New(WithError(originalErr))
			unwrapped := err.Unwrap()
			So(unwrapped, ShouldEqual, originalErr)
		})

		Convey("When unwrapping an error with multiple wrapped errors", func() {
			originalErr := errors.New("original error")
			secondErr := errors.New("second error")
			err := New(WithError(originalErr), WithError(secondErr))
			unwrapped := err.Unwrap()
			So(unwrapped, ShouldEqual, originalErr)
		})
	})
}

func TestIs(t *testing.T) {
	Convey("Given various error comparison scenarios", t, func() {
		originalErr := errors.New("original error")

		Convey("When comparing with a contained error", func() {
			err := New(WithError(originalErr))
			So(err.Is(originalErr), ShouldBeTrue)
		})

		Convey("When comparing with a different error", func() {
			err := New(WithError(originalErr))
			otherErr := errors.New("other error")
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
			originalErr := errors.New("original error")
			err := New(
				WithType(ValidationError),
				WithStatus(BadRequestStatus),
				WithError(originalErr),
				WithMessage("test message"),
			)
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, ValidationError)
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
			So(err.Error(), ShouldContainSubstring, "test message")
			So(err.Error(), ShouldContainSubstring, "original error")
		})
	})
}

func TestNotFound(t *testing.T) {
	Convey("Given the NotFound helper function", t, func() {
		originalErr := errors.New("original error")

		Convey("When creating an error with nil underlying error", func() {
			err := NotFound(nil, "resource not found")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, ResourceNotFoundError)
			So(err.Status(), ShouldEqual, http.StatusNotFound)
			So(err.Error(), ShouldContainSubstring, "resource not found")
		})

		Convey("When creating an error with an underlying error", func() {
			err := NotFound(originalErr, "resource not found")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, ResourceNotFoundError)
			So(err.Status(), ShouldEqual, http.StatusNotFound)
			So(err.Error(), ShouldContainSubstring, "resource not found")
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
		})

		Convey("When creating an error with multiple messages", func() {
			err := NotFound(originalErr, "resource", "not", "found")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "resource not found")
		})
	})
}

func TestBadRequest(t *testing.T) {
	Convey("Given the BadRequest helper function", t, func() {
		originalErr := errors.New("original error")

		Convey("When creating an error with nil underlying error", func() {
			err := BadRequest(nil, "invalid request")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, InvalidInputError)
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
			So(err.Error(), ShouldContainSubstring, "invalid request")
		})

		Convey("When creating an error with an underlying error", func() {
			err := BadRequest(originalErr, "invalid request")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, InvalidInputError)
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
			So(err.Error(), ShouldContainSubstring, "invalid request")
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
		})

		Convey("When creating an error with multiple messages", func() {
			err := BadRequest(originalErr, "invalid", "request", "format")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "invalid request format")
		})
	})
}

func TestInternalError(t *testing.T) {
	Convey("Given the InternalError helper function", t, func() {
		originalErr := errors.New("original error")

		Convey("When creating an error with nil underlying error", func() {
			err := InternalError(nil, "internal server error")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, SystemError)
			So(err.Status(), ShouldEqual, http.StatusInternalServerError)
			So(err.Error(), ShouldContainSubstring, "internal server error")
		})

		Convey("When creating an error with an underlying error", func() {
			err := InternalError(originalErr, "internal server error")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, SystemError)
			So(err.Status(), ShouldEqual, http.StatusInternalServerError)
			So(err.Error(), ShouldContainSubstring, "internal server error")
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
		})

		Convey("When creating an error with multiple messages", func() {
			err := InternalError(originalErr, "internal", "processing", "failed")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "internal processing failed")
		})
	})
}

func TestUnauthorized(t *testing.T) {
	Convey("Given the Unauthorized helper function", t, func() {
		originalErr := errors.New("original error")

		Convey("When creating an error with nil underlying error", func() {
			err := Unauthorized(nil, "unauthorized access")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, AuthenticationError)
			So(err.Status(), ShouldEqual, http.StatusUnauthorized)
			So(err.Error(), ShouldContainSubstring, "unauthorized access")
		})

		Convey("When creating an error with an underlying error", func() {
			err := Unauthorized(originalErr, "unauthorized access")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, AuthenticationError)
			So(err.Status(), ShouldEqual, http.StatusUnauthorized)
			So(err.Error(), ShouldContainSubstring, "unauthorized access")
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
		})

		Convey("When creating an error with multiple messages", func() {
			err := Unauthorized(originalErr, "invalid", "credentials", "provided")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "invalid credentials provided")
		})
	})
}

func TestForbidden(t *testing.T) {
	Convey("Given the Forbidden helper function", t, func() {
		originalErr := errors.New("original error")

		Convey("When creating an error with nil underlying error", func() {
			err := Forbidden(nil, "access forbidden")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, AuthorizationError)
			So(err.Status(), ShouldEqual, http.StatusForbidden)
			So(err.Error(), ShouldContainSubstring, "access forbidden")
		})

		Convey("When creating an error with an underlying error", func() {
			err := Forbidden(originalErr, "access forbidden")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, AuthorizationError)
			So(err.Status(), ShouldEqual, http.StatusForbidden)
			So(err.Error(), ShouldContainSubstring, "access forbidden")
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
		})

		Convey("When creating an error with multiple messages", func() {
			err := Forbidden(originalErr, "insufficient", "permissions", "for", "resource")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "insufficient permissions for resource")
		})
	})
}

func TestTimeout(t *testing.T) {
	Convey("Given the Timeout helper function", t, func() {
		originalErr := errors.New("original error")

		Convey("When creating an error with nil underlying error", func() {
			err := Timeout(nil, "request timeout")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, TimeoutError)
			So(err.Status(), ShouldEqual, http.StatusGatewayTimeout)
			So(err.Error(), ShouldContainSubstring, "request timeout")
		})

		Convey("When creating an error with an underlying error", func() {
			err := Timeout(originalErr, "request timeout")
			So(err, ShouldNotBeNil)
			So(err.Type(), ShouldEqual, TimeoutError)
			So(err.Status(), ShouldEqual, http.StatusGatewayTimeout)
			So(err.Error(), ShouldContainSubstring, "request timeout")
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
		})

		Convey("When creating an error with multiple messages", func() {
			err := Timeout(originalErr, "operation", "timed", "out")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "operation timed out")
		})
	})
}

func TestWrap(t *testing.T) {
	Convey("Given various error wrapping scenarios", t, func() {
		Convey("When wrapping nil error", func() {
			err := Wrap(nil, "wrapped message")
			So(err, ShouldBeNil)
		})

		Convey("When wrapping an error with a message", func() {
			originalErr := errors.New("original error")
			err := Wrap(originalErr, "wrapped message")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "wrapped message")
			So(err.Error(), ShouldContainSubstring, "original error")
		})

		Convey("When wrapping an error with a formatted message", func() {
			originalErr := errors.New("original error")
			err := Wrap(originalErr, "wrapped message: %v", "detail")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "wrapped message: detail")
			So(err.Error(), ShouldContainSubstring, "original error")
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

		Convey("When using with other options", func() {
			err := New(
				WithType(ValidationError),
				WithMessage("test message"),
			)
			So(err.Type(), ShouldEqual, ValidationError)
			So(err.Error(), ShouldContainSubstring, "test message")
		})
	})
}

func TestWithError(t *testing.T) {
	Convey("Given the WithError option function", t, func() {
		originalErr := errors.New("original error")

		Convey("When adding a nil error", func() {
			err := New(WithError(nil))
			So(len(err.errors), ShouldEqual, 0)
		})

		Convey("When adding a valid error", func() {
			err := New(WithError(originalErr))
			So(len(err.errors), ShouldEqual, 1)
			So(err.errors[0], ShouldEqual, originalErr)
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
		})

		Convey("When adding multiple errors", func() {
			secondErr := errors.New("second error")
			err := New(
				WithError(originalErr),
				WithError(secondErr),
			)
			So(len(err.errors), ShouldEqual, 2)
			So(err.errors[0], ShouldEqual, originalErr)
			So(err.errors[1], ShouldEqual, secondErr)
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
			So(err.Error(), ShouldContainSubstring, secondErr.Error())
		})

		Convey("When using with other options", func() {
			err := New(
				WithError(originalErr),
				WithMessage("test message"),
			)
			So(len(err.errors), ShouldEqual, 1)
			So(err.Error(), ShouldContainSubstring, "test message")
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
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
			message := "test message"
			err := New(WithMessage(message))
			So(len(err.messages), ShouldEqual, 1)
			So(err.messages[0], ShouldEqual, message)
			So(err.Error(), ShouldContainSubstring, message)
		})

		Convey("When adding multiple messages", func() {
			firstMsg := "first message"
			secondMsg := "second message"
			err := New(
				WithMessage(firstMsg),
				WithMessage(secondMsg),
			)
			So(len(err.messages), ShouldEqual, 2)
			So(err.messages[0], ShouldEqual, firstMsg)
			So(err.messages[1], ShouldEqual, secondMsg)
			So(err.Error(), ShouldContainSubstring, firstMsg)
			So(err.Error(), ShouldContainSubstring, secondMsg)
		})

		Convey("When using with other options", func() {
			originalErr := errors.New("original error")
			message := "test message"
			err := New(
				WithMessage(message),
				WithError(originalErr),
			)
			So(len(err.messages), ShouldEqual, 1)
			So(err.Error(), ShouldContainSubstring, message)
			So(err.Error(), ShouldContainSubstring, originalErr.Error())
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

		Convey("When using with other options", func() {
			err := New(
				WithStatus(BadRequestStatus),
				WithMessage("test message"),
				WithType(ValidationError),
			)
			So(err.Status(), ShouldEqual, http.StatusBadRequest)
			So(err.Type(), ShouldEqual, ValidationError)
			So(err.Error(), ShouldContainSubstring, "test message")
		})

		Convey("When using an undefined status", func() {
			err := New(WithStatus(ErrnieStatusType(999)))
			So(err.Status(), ShouldEqual, http.StatusInternalServerError)
		})
	})
}
