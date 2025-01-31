package stream

import (
	"context"
	"errors"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/provider"
)

func TestNewAccumulator(t *testing.T) {
	Convey("Given a call to NewAccumulator", t, func() {
		accumulator := NewAccumulator()

		Convey("Then it should be properly initialized", func() {
			So(accumulator, ShouldNotBeNil)
			So(accumulator.wg, ShouldNotBeNil)
			So(accumulator.chunks, ShouldBeEmpty)
			So(accumulator.err, ShouldBeNil)
		})
	})
}

func TestGenerate(t *testing.T) {
	Convey("Given an Accumulator instance", t, func() {
		accumulator := NewAccumulator()
		ctx := context.Background()
		in := make(chan *provider.Event)
		out := accumulator.Generate(ctx, in)

		Convey("When sending normal text events", func() {
			go func() {
				event := provider.NewEvent("generate:contentblock:delta", provider.EventChunk, "test text", "", nil)
				in <- event
				close(in)
			}()

			var result *provider.Event
			for event := range out {
				result = event
			}

			Convey("Then it should process events correctly", func() {
				So(result, ShouldNotBeNil)
				So(result.Type, ShouldEqual, provider.EventChunk)
				So(result.Text, ShouldEqual, "test text")
			})
		})

		Convey("When sending error events", func() {
			go func() {
				event := provider.NewEvent("generate:error", provider.EventError, "test error", "", nil)
				in <- event
				close(in)
			}()

			var result *provider.Event
			for event := range out {
				result = event
			}

			Convey("Then it should handle errors correctly", func() {
				So(result, ShouldNotBeNil)
				So(result.Type, ShouldEqual, provider.EventError)
				So(result.Text, ShouldEqual, "test error")
			})
		})
	})
}

func TestWait(t *testing.T) {
	Convey("Given an Accumulator instance", t, func() {
		accumulator := NewAccumulator()
		ctx := context.Background()
		in := make(chan *provider.Event)
		out := accumulator.Generate(ctx, in)

		Convey("When waiting for completion", func() {
			go func() {
				event := provider.NewEvent("generate:contentblock:delta", provider.EventChunk, "test text", "", nil)
				in <- event
				close(in)
			}()

			for range out {
				// Consume events
			}
			accumulator.Wait()

			Convey("Then it should complete without deadlock", func() {
				So(true, ShouldBeTrue) // If we reach here, Wait() completed successfully
			})
		})
	})
}

func TestString(t *testing.T) {
	Convey("Given an Accumulator instance", t, func() {
		accumulator := NewAccumulator()

		Convey("When getting string output", func() {
			accumulator.Write([]byte(" test string "))
			result := accumulator.String()

			Convey("Then it should return trimmed string", func() {
				So(result, ShouldEqual, "test string")
			})
		})
	})
}

func TestCompile(t *testing.T) {
	Convey("Given an Accumulator instance", t, func() {
		accumulator := NewAccumulator()

		Convey("When compiling normal chunks", func() {
			accumulator.Write([]byte("chunk1 "))
			accumulator.Write([]byte("chunk2"))
			result := accumulator.Compile()

			Convey("Then it should combine chunks correctly", func() {
				So(result, ShouldNotBeNil)
				So(result.Type, ShouldEqual, "generate:contentblock:done")
				So(result.Text, ShouldEqual, "chunk1 chunk2")
			})
		})

		Convey("When compiling partial JSON chunks", func() {
			event := provider.NewEvent("generate:contentblock:delta", provider.EventChunk, `{"key":"value"}`, "", nil)
			accumulator.chunks = append(accumulator.chunks, event)
			result := accumulator.Compile()

			Convey("Then it should handle partial JSON", func() {
				So(result, ShouldNotBeNil)
				So(result.Type, ShouldEqual, "generate:contentblock:done")
				So(result.Text, ShouldEqual, `{"key":"value"}`)
			})
		})

		Convey("When compiling with error state", func() {
			accumulator.err = errors.New("test error")
			result := accumulator.Compile()

			Convey("Then it should return error event", func() {
				So(result, ShouldNotBeNil)
				So(result.Type, ShouldEqual, provider.EventError)
				So(result.Text, ShouldEqual, "test error")
			})
		})
	})
}

func TestError(t *testing.T) {
	Convey("Given an Accumulator instance", t, func() {
		accumulator := NewAccumulator()

		Convey("When checking error state", func() {
			testErr := errors.New("test error")
			accumulator.err = testErr

			Convey("Then it should return the error", func() {
				So(accumulator.Error(), ShouldEqual, testErr)
			})
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given an Accumulator instance", t, func() {
		accumulator := NewAccumulator()

		Convey("When writing text", func() {
			text := []byte("test text")
			result := accumulator.Write(text)

			Convey("Then it should append to chunks", func() {
				So(result, ShouldEqual, accumulator)
				So(len(accumulator.chunks), ShouldEqual, 1)
				So(accumulator.chunks[0].Text, ShouldEqual, "test text")
			})
		})
	})
}
