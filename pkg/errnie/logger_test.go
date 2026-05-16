package errnie

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/phuslu/log"
	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/config"
)

func TestApply(t *testing.T) {
	Convey("Given Apply", t, func() {
		saved := log.DefaultLogger
		defer func() {
			log.DefaultLogger = saved
		}()

		Convey("When level is debug", func() {
			Apply(&config.ErrnieConfig{Level: "debug"})
			So(log.DefaultLogger.Level, ShouldEqual, log.DebugLevel)
		})

		Convey("When level is trace", func() {
			Apply(&config.ErrnieConfig{Level: "trace"})
			So(log.DefaultLogger.Level, ShouldEqual, log.TraceLevel)
		})

		Convey("When level is warn", func() {
			Apply(&config.ErrnieConfig{Level: "warn"})
			So(log.DefaultLogger.Level, ShouldEqual, log.WarnLevel)
		})

		Convey("When level is warning alias", func() {
			Apply(&config.ErrnieConfig{Level: "warning"})
			So(log.DefaultLogger.Level, ShouldEqual, log.WarnLevel)
		})

		Convey("When level is error", func() {
			Apply(&config.ErrnieConfig{Level: "error"})
			So(log.DefaultLogger.Level, ShouldEqual, log.ErrorLevel)
		})

		Convey("When level is fatal", func() {
			Apply(&config.ErrnieConfig{Level: "fatal"})
			So(log.DefaultLogger.Level, ShouldEqual, log.FatalLevel)
		})

		Convey("When level is panic", func() {
			Apply(&config.ErrnieConfig{Level: "panic"})
			So(log.DefaultLogger.Level, ShouldEqual, log.PanicLevel)
		})

		Convey("When level is empty it should default to info", func() {
			Apply(&config.ErrnieConfig{Level: ""})
			So(log.DefaultLogger.Level, ShouldEqual, log.InfoLevel)
		})

		Convey("When level is unknown it should default to info", func() {
			Apply(&config.ErrnieConfig{Level: "not-a-level"})
			So(log.DefaultLogger.Level, ShouldEqual, log.InfoLevel)
		})

		Convey("When file sink is active it should include file writer", func() {
			dir := t.TempDir()
			path := filepath.Join(dir, "errnie.log")

			Apply(&config.ErrnieConfig{
				Level: "info",
				File: config.FileConfig{
					Active: true,
					Path:   path,
				},
			})

			Info("file-sink-line")

			waitFor(func() bool {
				body, readErr := os.ReadFile(path)
				return readErr == nil && strings.Contains(string(body), "file-sink-line")
			}, 500*time.Millisecond)

			body, readErr := os.ReadFile(path)
			So(readErr, ShouldBeNil)
			So(string(body), ShouldContainSubstring, "file-sink-line")
		})

		Convey("When elasticsearch sink is active it should POST log lines", func() {
			var mu sync.Mutex
			var bodies []string

			server := newElasticsearchLogServer(t, func(body string) {
				mu.Lock()
				bodies = append(bodies, body)
				mu.Unlock()
			})
			defer server.Close()

			Apply(&config.ErrnieConfig{
				Level: "info",
				Elasticsearch: config.ElasticsearchConfig{
					Active: true,
					URL:    server.URL,
					Index:  "caramba",
				},
			})

			Info("elastic-sink-line")

			waitFor(func() bool {
				mu.Lock()
				ok := len(bodies) > 0 && strings.Contains(bodies[len(bodies)-1], "elastic-sink-line")
				mu.Unlock()

				return ok
			}, 3*time.Second)

			mu.Lock()
			got := strings.Join(bodies, "\n")
			mu.Unlock()

			So(got, ShouldContainSubstring, "elastic-sink-line")
		})
	})
}

func TestNewLogger(t *testing.T) {
	Convey("Given NewLogger", t, func() {
		Convey("It should reference the default logger handle", func() {
			handle := NewLogger()
			So(handle, ShouldNotBeNil)
			So(handle.handle, ShouldEqual, &log.DefaultLogger)
		})
	})
}

func TestError(t *testing.T) {
	Convey("Given Error", t, func() {
		out := captureStdout(t, func() {
			Apply(&config.ErrnieConfig{Level: "error"})
			So(Error(nil, "k", "v"), ShouldBeNil)

			base := errors.New("boom")
			So(Error(base, "ctx", "unit"), ShouldEqual, base)
		})

		So(out, ShouldContainSubstring, "boom")
	})
}

func TestWarn(t *testing.T) {
	Convey("Given Warn", t, func() {
		out := captureStdout(t, func() {
			Apply(&config.ErrnieConfig{Level: "warn"})
			Warn("warn-message", "rid", "42")
		})

		So(out, ShouldContainSubstring, "warn-message")
	})
}

func TestInfo(t *testing.T) {
	Convey("Given Info", t, func() {
		out := captureStdout(t, func() {
			Apply(&config.ErrnieConfig{Level: "info"})
			Info("info-message", "env", "test")
		})

		So(out, ShouldContainSubstring, "info-message")
	})
}

func TestDebug(t *testing.T) {
	Convey("Given Debug", t, func() {
		out := captureStdout(t, func() {
			Apply(&config.ErrnieConfig{Level: "debug"})
			Debug("debug-message", "span", "x")
		})

		So(out, ShouldContainSubstring, "debug-message")
	})
}

func TestTrace(t *testing.T) {
	Convey("Given Trace", t, func() {
		out := captureStdout(t, func() {
			Apply(&config.ErrnieConfig{Level: "trace"})
			Trace("trace-message", "tid", "y")
		})

		So(out, ShouldContainSubstring, "trace-message")
	})
}

func TestSuppressLogging(t *testing.T) {
	Convey("Given SuppressLogging", t, func() {
		out := captureStdout(t, func() {
			Apply(&config.ErrnieConfig{Level: "debug"})
			restore := SuppressLogging()

			Info("hidden-info")
			Warn("hidden-warn")
			Debug("hidden-debug")
			So(Error(errors.New("hidden-error")), ShouldNotBeNil)
			restore()
			restore()
			Info("visible-info")
		})

		So(out, ShouldNotContainSubstring, "hidden")
		So(out, ShouldContainSubstring, "visible-info")
	})
}

func BenchmarkApply(b *testing.B) {
	saved := log.DefaultLogger
	defer func() {
		log.DefaultLogger = saved
	}()

	cfg := &config.ErrnieConfig{Level: "info"}

	for range b.N {
		Apply(cfg)
	}
}

func captureStdout(testing *testing.T, fn func()) string {
	testing.Helper()

	reader, writer, pipeErr := os.Pipe()
	So(pipeErr, ShouldBeNil)

	original := os.Stdout
	os.Stdout = writer

	fn()

	closeWriterErr := writer.Close()
	So(closeWriterErr, ShouldBeNil)

	os.Stdout = original

	buf, readErr := io.ReadAll(reader)

	closeReaderErr := reader.Close()
	So(closeReaderErr, ShouldBeNil)
	So(readErr, ShouldBeNil)

	return string(buf)
}

func waitFor(predicate func() bool, deadline time.Duration) {
	deadlineAt := time.Now().Add(deadline)

	for time.Now().Before(deadlineAt) {
		if predicate() {
			return
		}

		time.Sleep(5 * time.Millisecond)
	}
}

func newElasticsearchLogServer(testing *testing.T, onBody func(body string)) *httptest.Server {
	testing.Helper()

	return httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		payload, readErr := io.ReadAll(request.Body)

		if readErr != nil {
			response.WriteHeader(http.StatusInternalServerError)

			return
		}

		onBody(string(payload))
		response.WriteHeader(http.StatusCreated)
	}))
}
