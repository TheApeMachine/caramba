package errnie

import (
	"errors"
	"fmt"
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
			err := RunSafelyErr(func() error {
				return errTest
			})
			So(err, ShouldNotBeNil)
			So(err, ShouldEqual, errTest)
		})

		Convey("When the function panics with an *ErrnieError", func() {
			err := RunSafelyErr(func() error {
				panic(errnieTest)
				// Unreachable but needed for compiler
				// return nil
			})
			So(err, ShouldNotBeNil)
			So(err, ShouldHaveSameTypeAs, &ErrnieError{})
			So(err.Error(), ShouldEqual, errnieTest.Error())
		})

		Convey("When the function panics with a different value", func() {
			panicValue := "regular panic"
			So(func() {
				RunSafelyErr(func() error {
					panic(panicValue)
					// return nil
				})
			}, ShouldPanicWith, panicValue)
		})

		Convey("When the function panics with a standard error", func() {
			panicValue := errors.New("standard error panic")
			So(func() {
				RunSafelyErr(func() error {
					panic(panicValue)
					// return nil
				})
			}, ShouldPanicWith, panicValue)
		})
	})
}

func TestJumpReturn(t *testing.T) {
	Convey("Test JumpReturn", t, func() {
		testValue := "test value"

		Convey("When error is nil", func() {
			result := JumpReturn(testValue, nil)
			So(result, ShouldEqual, testValue)
		})

		Convey("When error is a standard error", func() {
			var recoveredValue any
			func() {
				defer func() {
					recoveredValue = recover()
				}()
				_ = JumpReturn(testValue, errTest)
			}()

			So(recoveredValue, ShouldNotBeNil)
			ret, ok := recoveredValue.(Return)
			So(ok, ShouldBeTrue)
			So(ret.Value, ShouldEqual, testValue)
			So(ret.Error, ShouldNotBeNil)
			So(ret.Error.Type(), ShouldEqual, SystemError) // Default wrapper type
			So(ret.Error.Error(), ShouldContainSubstring, errTest.Error())
		})

		Convey("When error is an *ErrnieError", func() {
			var recoveredValue any
			func() {
				defer func() {
					recoveredValue = recover()
				}()
				_ = JumpReturn(testValue, errnieTest)
			}()

			So(recoveredValue, ShouldNotBeNil)
			ret, ok := recoveredValue.(Return)
			So(ok, ShouldBeTrue)
			So(ret.Value, ShouldEqual, testValue)
			So(ret.Error, ShouldEqual, errnieTest) // Should be the original *ErrnieError
		})
	})
}

func TestSafe(t *testing.T) {
	Convey("Test Safe", t, func() {
		successValue := "success"
		panicValue := "other panic"

		Convey("When the function executes successfully", func() {
			result := Safe(func() string {
				return successValue
			})
			So(result, ShouldEqual, successValue)
		})

		Convey("When the function panics with Return (via JumpReturn)", func() {
			var recoveredErr error
			func() {
				defer func() {
					if r := recover(); r != nil {
						if ee, ok := r.(*ErrnieError); ok {
							recoveredErr = ee
						} else {
							panic(fmt.Sprintf("Expected *ErrnieError panic, got %T", r))
						}
					}
				}()
				Safe(func() string {
					return JumpReturn(successValue, errnieTest)
				})
				// Should not reach here
				panic("should have panicked")
			}()
			So(recoveredErr, ShouldNotBeNil)
			So(recoveredErr, ShouldEqual, errnieTest)
		})

		Convey("When the function panics with a non-Return value", func() {
			So(func() {
				Safe(func() string {
					panic(panicValue)
				})
			}, ShouldPanicWith, panicValue)
		})

		Convey("When the function panics with a non-Return struct", func() {
			type NonReturn struct {
				Val int
			}
			customPanic := NonReturn{Val: 1}
			So(func() {
				Safe(func() string {
					panic(customPanic)
				})
			}, ShouldPanicWith, customPanic)
		})
	})
}

func TestTry(t *testing.T) {
	Convey("Test Try", t, func() {
		successValue := "success"
		fnSuccess := func() (string, error) {
			return successValue, nil
		}
		fnFailStd := func() (string, error) {
			return successValue, errTest
		}
		fnFailErrnie := func() (string, error) {
			return successValue, errnieTest
		}

		Convey("When the function returns no error", func() {
			// We need RunSafely here because Try itself doesn't recover,
			// it relies on a higher-level recovery mechanism.
			var result string
			err := RunSafely(func() {
				result = Try(fnSuccess())
			})
			So(err, ShouldBeNil)
			So(result, ShouldEqual, successValue)
		})

		Convey("When the function returns a standard error", func() {
			var result string
			err := RunSafely(func() {
				// This call will panic
				result = Try(fnFailStd())
				// Should not reach here
				t.Error("Try should have panicked")
			})

			So(err, ShouldNotBeNil)
			So(result, ShouldBeEmpty) // result should not be assigned
			ee, ok := err.(*ErrnieError)
			So(ok, ShouldBeTrue)
			So(ee.Type(), ShouldEqual, SystemError) // Default wrapper type
			So(ee.Error(), ShouldContainSubstring, errTest.Error())
		})

		Convey("When the function returns an *ErrnieError", func() {
			var result string
			err := RunSafely(func() {
				// This call will panic
				result = Try(fnFailErrnie())
				// Should not reach here
				t.Error("Try should have panicked")
			})

			So(err, ShouldNotBeNil)
			So(result, ShouldBeEmpty) // result should not be assigned
			ee, ok := err.(*ErrnieError)
			So(ok, ShouldBeTrue)
			So(ee, ShouldEqual, errnieTest) // Should be the original *ErrnieError
		})
	})
}
