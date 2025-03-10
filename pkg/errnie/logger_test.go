package errnie

import (
	"errors"
	"os"
	"testing"

	"github.com/charmbracelet/log"
	. "github.com/smartystreets/goconvey/convey"
)

// TestSetLevel tests the SetLevel function
func TestSetLevel(t *testing.T) {
	Convey("Given a log level", t, func() {
		// Save the original log level to restore it later
		originalLevel := logger.GetLevel()
		defer logger.SetLevel(originalLevel)

		Convey("When setting the log level to DEBUG", func() {
			SetLevel(log.DebugLevel)

			Convey("Then the logger's level should be updated", func() {
				So(logger.GetLevel(), ShouldEqual, log.DebugLevel)
			})
		})

		Convey("When setting the log level to ERROR", func() {
			SetLevel(log.ErrorLevel)

			Convey("Then the logger's level should be updated", func() {
				So(logger.GetLevel(), ShouldEqual, log.ErrorLevel)
			})
		})
	})
}

// TestError tests the Error function
func TestError(t *testing.T) {
	Convey("Given an error message and key-value pairs", t, func() {
		// Temporarily redirect stderr to capture output
		originalStderr := os.Stderr
		_, w, _ := os.Pipe()
		os.Stderr = w

		defer func() {
			os.Stderr = originalStderr
		}()

		testErr := errors.New("test error")

		Convey("When calling Error with nil message", func() {
			result := Error(nil)

			Convey("Then it should return nil", func() {
				So(result, ShouldBeNil)
			})
		})

		Convey("When calling Error with an error in keyvals", func() {
			result := Error("Error occurred", "err", testErr)

			Convey("Then it should return the error", func() {
				So(result, ShouldEqual, testErr)
			})

			// Close the writer to get the stderr output
			w.Close()

			// We don't check the exact output since it depends on the logger format
			// Just checking that the function completes
		})
	})
}

// TestDebug tests the Debug function
func TestDebug(t *testing.T) {
	Convey("Given a debug message and key-value pairs", t, func() {
		// Temporarily redirect stderr to capture output
		originalStderr := os.Stderr
		_, w, _ := os.Pipe()
		os.Stderr = w

		// Save the original log level to restore it later
		originalLevel := logger.GetLevel()
		defer func() {
			os.Stderr = originalStderr
			logger.SetLevel(originalLevel)
		}()

		// Set the log level to debug to ensure messages are logged
		logger.SetLevel(log.DebugLevel)

		Convey("When calling Debug with a message", func() {
			Debug("Debug message", "key", "value")

			// Close the writer to get the stderr output
			w.Close()

			// We don't check the exact output since it depends on the logger format
			// Just checking that the function completes without panicking
		})
	})
}

// TestGetStackTrace tests the getStackTrace function
func TestGetStackTrace(t *testing.T) {
	Convey("When getting a stack trace", t, func() {
		trace := getStackTrace()

		Convey("Then it should return a non-empty string", func() {
			So(trace, ShouldNotBeBlank)
			So(trace, ShouldContainSubstring, "STACK TRACE")
		})
	})
}

// TestGetCodeSnippet tests the getCodeSnippet function
func TestGetCodeSnippet(t *testing.T) {
	Convey("Given a file path and line number", t, func() {
		Convey("When the file exists", func() {
			// Use this test file itself as the file to get a snippet from
			snippet := getCodeSnippet("logger_test.go", 1, 2)

			Convey("Then it should return a non-empty snippet", func() {
				// The exact content may vary, so we just check that it's not empty
				// and has the right format
				So(snippet, ShouldNotBeBlank)
			})
		})

		Convey("When the file doesn't exist", func() {
			snippet := getCodeSnippet("nonexistent_file.go", 1, 2)

			Convey("Then it should return an empty string", func() {
				So(snippet, ShouldBeBlank)
			})
		})

		Convey("When an empty file path is provided", func() {
			snippet := getCodeSnippet("", 1, 2)

			Convey("Then it should return an empty string", func() {
				So(snippet, ShouldBeBlank)
			})
		})
	})
}
