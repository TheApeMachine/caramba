package errnie

import (
	"errors"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewError tests the NewError function
func TestNewError(t *testing.T) {
	Convey("Given an error", t, func() {
		testErr := errors.New("test error")

		Convey("When creating a new error using NewError", func() {
			err := NewError(testErr)

			Convey("Then it should wrap the error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "test error")
			})
		})

		Convey("When passing nil to NewError", func() {
			err := NewError(nil)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}

// TestErrnieErrorError tests the Error method of ErrnieError
func TestErrnieErrorError(t *testing.T) {
	Convey("Given an ErrnieError", t, func() {
		errObj := &ErrnieError{msg: "test message"}

		Convey("When calling Error()", func() {
			errMsg := errObj.Error()

			Convey("Then it should return the error message", func() {
				So(errMsg, ShouldEqual, "test message")
			})
		})
	})
}

// TestNewErrIO tests the NewErrIO function
func TestNewErrIO(t *testing.T) {
	Convey("Given an IO error", t, func() {
		testErr := errors.New("io test error")

		Convey("When creating a new IO error", func() {
			err := NewErrIO(testErr)

			Convey("Then it should return a wrapped error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "io test error")
			})
		})

		Convey("When passing nil to NewErrIO", func() {
			err := NewErrIO(nil)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When passing EOF to NewErrIO", func() {
			err := NewErrIO(io.EOF)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}

// TestErrIOError tests the Error method of ErrIO
func TestErrIOError(t *testing.T) {
	Convey("Given an ErrIO", t, func() {
		innerErr := errors.New("inner io error")
		errObj := ErrIO{Err: innerErr}

		Convey("When calling Error()", func() {
			errMsg := errObj.Error()

			Convey("Then it should return the inner error message", func() {
				So(errMsg, ShouldEqual, "inner io error")
			})
		})
	})
}

// TestNewErrValidation tests the NewErrValidation function
func TestNewErrValidation(t *testing.T) {
	Convey("Given validation error messages", t, func() {
		Convey("When creating a validation error with one message", func() {
			err := NewErrValidation("field is required")

			Convey("Then it should return a wrapped error with message", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "field is required")
			})
		})

		Convey("When creating a validation error with multiple messages", func() {
			err := NewErrValidation("field", "is", "required")

			Convey("Then it should join the messages", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "field is required")
			})
		})

		Convey("When creating a validation error with no messages", func() {
			err := NewErrValidation()

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}

// TestErrValidationError tests the Error method of ErrValidation
func TestErrValidationError(t *testing.T) {
	Convey("Given an ErrValidation", t, func() {
		innerErr := errors.New("validation error")
		errObj := ErrValidation{Err: innerErr}

		Convey("When calling Error()", func() {
			errMsg := errObj.Error()

			Convey("Then it should return the inner error message", func() {
				So(errMsg, ShouldEqual, "validation error")
			})
		})
	})
}

// TestNewErrParse tests the NewErrParse function
func TestNewErrParse(t *testing.T) {
	Convey("Given a parse error", t, func() {
		testErr := errors.New("parse test error")

		Convey("When creating a new parse error", func() {
			err := NewErrParse(testErr)

			Convey("Then it should return a wrapped error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "parse test error")
			})
		})

		Convey("When passing nil to NewErrParse", func() {
			err := NewErrParse(nil)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When passing EOF to NewErrParse", func() {
			err := NewErrParse(io.EOF)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}

// TestErrParseError tests the Error method of ErrParse
func TestErrParseError(t *testing.T) {
	Convey("Given an ErrParse", t, func() {
		innerErr := errors.New("parse error")
		errObj := ErrParse{Err: innerErr}

		Convey("When calling Error()", func() {
			errMsg := errObj.Error()

			Convey("Then it should return the inner error message", func() {
				So(errMsg, ShouldEqual, "parse error")
			})
		})
	})
}

// TestNewErrOperation tests the NewErrOperation function
func TestNewErrOperation(t *testing.T) {
	Convey("Given an operation error", t, func() {
		testErr := errors.New("operation test error")

		Convey("When creating a new operation error", func() {
			err := NewErrOperation(testErr)

			Convey("Then it should return a wrapped error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "operation test error")
			})
		})

		Convey("When passing nil to NewErrOperation", func() {
			err := NewErrOperation(nil)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When passing EOF to NewErrOperation", func() {
			err := NewErrOperation(io.EOF)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}

// TestErrOperationError tests the Error method of ErrOperation
func TestErrOperationError(t *testing.T) {
	Convey("Given an ErrOperation", t, func() {
		innerErr := errors.New("operation error")
		errObj := ErrOperation{Err: innerErr}

		Convey("When calling Error()", func() {
			errMsg := errObj.Error()

			Convey("Then it should return the inner error message", func() {
				So(errMsg, ShouldEqual, "operation error")
			})
		})
	})
}

// TestNewErrHTTP tests the NewErrHTTP function
func TestNewErrHTTP(t *testing.T) {
	Convey("Given an HTTP error", t, func() {
		testErr := errors.New("http test error")

		Convey("When creating a new HTTP error with code 404", func() {
			err := NewErrHTTP(testErr, 404)

			Convey("Then it should return a wrapped error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldEqual, "Not Found: http test error")
			})
		})

		Convey("When passing nil to NewErrHTTP", func() {
			err := NewErrHTTP(nil, 500)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When passing EOF to NewErrHTTP", func() {
			err := NewErrHTTP(io.EOF, 500)

			Convey("Then it should return nil", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}

// TestErrHTTPError tests the Error method of ErrHTTP
func TestErrHTTPError(t *testing.T) {
	Convey("Given an ErrHTTP", t, func() {
		innerErr := errors.New("http error")

		Convey("When error code is 400", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 400}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Bad Request: http error")
			})
		})

		Convey("When error code is 401", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 401}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Unauthorized: http error")
			})
		})

		Convey("When error code is 403", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 403}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Forbidden: http error")
			})
		})

		Convey("When error code is 404", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 404}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Not Found: http error")
			})
		})

		Convey("When error code is 429", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 429}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Too Many Requests: http error")
			})
		})

		Convey("When error code is 500", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 500}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Internal Server Error: http error")
			})
		})

		Convey("When error code is 502", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 502}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Bad Gateway: http error")
			})
		})

		Convey("When error code is 503", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 503}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Service Unavailable: http error")
			})
		})

		Convey("When error code is 504", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 504}
			errMsg := errObj.Error()

			Convey("Then it should return formatted error message", func() {
				So(errMsg, ShouldEqual, "Gateway Timeout: http error")
			})
		})

		Convey("When error code is not standard (599)", func() {
			errObj := ErrHTTP{Err: innerErr, Code: 599}
			errMsg := errObj.Error()

			Convey("Then it should return generic formatted error message", func() {
				So(errMsg, ShouldEqual, "HTTP Error: 599 http error")
			})
		})
	})
}
