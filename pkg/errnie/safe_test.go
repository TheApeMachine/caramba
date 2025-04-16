package errnie

import (
	"errors"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

var errTest = errors.New("test error")
var errnieTest = New(WithMessage("errnie test error"))

func TestRunSafely(t *testing.T) {
	Convey("Test RunSafely", t, func() {
		Convey("When the function executes without panic", func() {
			err := RunSafely(func() {
				// No panic
			})
			So(err, ShouldBeNil)
		})

		Convey("When the function panics with an *ErrnieError", func() {
			err := RunSafely(func() {
				panic(errnieTest)
			})
			So(err, ShouldNotBeNil)
			So(err, ShouldHaveSameTypeAs, &ErrnieError{})
			So(err.Error(), ShouldEqual, errnieTest.Error())
		})

		Convey("When the function panics with a different value", func() {
			panicValue := "regular panic"
			So(func() {
				RunSafely(func() {
					panic(panicValue)
				})
			}, ShouldPanicWith, panicValue)
		})

		Convey("When the function panics with a standard error", func() {
			panicValue := errors.New("standard error panic")
			So(func() {
				RunSafely(func() {
					panic(panicValue)
				})
			}, ShouldPanicWith, panicValue)
		})
	})
}

func TestRunSafelyErr(t *testing.T) {
	Convey("Test RunSafelyErr", t, func() {
		Convey("When the function returns nil error and no panic", func() {
			err := RunSafelyErr(func() error {
				return nil
			})
			So(err, ShouldBeNil)
		})

		Convey("When the function returns a non-nil error and no panic", func() {
			testErr := errors.New("test error")
			err := RunSafelyErr(func() error {
				return testErr
			})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "test error")
			So(err.(*ErrnieError), ShouldNotBeNil)
		})

		Convey("When the function panics with an *ErrnieError", func() {
			err := RunSafelyErr(func() error {
				panic(New(WithError(errors.New("test panic")), WithMessage("test panic")))
			})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "test panic")
		})

		Convey("When the function panics with a different value", func() {
			So(func() {
				RunSafelyErr(func() error {
					panic("different panic")
				})
			}, ShouldPanic)
		})

		Convey("When the function panics with a standard error", func() {
			So(func() {
				RunSafelyErr(func() error {
					panic(errors.New("standard error"))
				})
			}, ShouldPanic)
		})
	})
}

func TestJumpReturn(t *testing.T) {
	Convey("Test JumpReturn", t, func() {
		Convey("When error is nil", func() {
			result := JumpReturn("test", nil)
			So(result, ShouldEqual, "test")
		})

		Convey("When error is a standard error", func() {
			testErr := errors.New("test error")
			So(func() {
				JumpReturn("test", testErr)
			}, ShouldPanic)
		})

		Convey("When error is an *ErrnieError", func() {
			errnie := New(WithMessage("test error"))
			So(func() {
				JumpReturn("test", errnie)
			}, ShouldPanic)
		})
	})
}

func TestSafe(t *testing.T) {
	Convey("Test Safe", t, func() {
		Convey("When the function executes successfully", func() {
			result := Safe(func() string {
				return "success"
			})
			So(result, ShouldEqual, "success")
		})

		Convey("When the function panics with Return (via JumpReturn)", func() {
			So(func() {
				Safe(func() string {
					JumpReturn("test", errors.New("test error"))
					return "unreachable"
				})
			}, ShouldPanic)
		})

		Convey("When the function panics with a non-Return value", func() {
			So(func() {
				Safe(func() string {
					panic("different panic")
					return "unreachable"
				})
			}, ShouldPanic)
		})

		Convey("When the function panics with a non-Return struct", func() {
			So(func() {
				Safe(func() string {
					panic(struct{ msg string }{"test"})
					return "unreachable"
				})
			}, ShouldPanic)
		})
	})
}

func TestTry(t *testing.T) {
	Convey("Test Try", t, func() {
		Convey("When the function returns no error", func() {
			result := RunSafely(func() {
				val := Try("success", nil)
				So(val, ShouldEqual, "success")
			})
			So(result, ShouldBeNil)
		})

		Convey("When the function returns a standard error", func() {
			err := RunSafely(func() {
				Try("test", errors.New("test error"))
			})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "test error")
			So(err.(*ErrnieError).Type(), ShouldEqual, SystemError)
		})

		Convey("When the function returns an *ErrnieError", func() {
			errnie := New(WithMessage("test error"))
			err := RunSafely(func() {
				Try("test", errnie)
			})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "test error")
		})
	})
}
