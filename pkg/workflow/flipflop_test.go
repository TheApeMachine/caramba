package workflow

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/stream"
)

func TestFlipFlop(t *testing.T) {
	Convey("Given a FlipFlop function", t, func() {
		Convey("When using it to modify artifact state through a process", func() {
			// Create a setter that modifies artifact metadata
			setter := NewTestSetter()

			// Create an empty artifact
			artifact := datura.New()

			// Initial state should be empty
			initialOutput := datura.GetMetaValue[string](artifact, "output")
			So(initialOutput, ShouldEqual, "")

			// Flip artifact into setter and flop back
			err := NewFlipFlop(artifact, setter)

			Convey("Then it should succeed", func() {
				So(err, ShouldBeNil)
			})

			Convey("Then the artifact state should be modified", func() {
				output := datura.GetMetaValue[string](artifact, "output")
				So(output, ShouldEqual, "hello")
			})
		})

		Convey("When using it with a setter that adds multiple metadata values", func() {
			// Create a more complex setter
			setter := NewComplexSetter()
			artifact := datura.New()

			err := NewFlipFlop(artifact, setter)
			So(err, ShouldBeNil)

			Convey("Then all metadata should be set correctly", func() {
				So(datura.GetMetaValue[string](artifact, "name"), ShouldEqual, "test")
				So(datura.GetMetaValue[int](artifact, "count"), ShouldEqual, 42)
				So(datura.GetMetaValue[bool](artifact, "processed"), ShouldBeTrue)
			})
		})
	})
}

// TestSetter implements the example from the FlipFlop documentation
type TestSetter struct {
	buffer *stream.Buffer
}

func NewTestSetter() *TestSetter {
	return &TestSetter{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			// Set the output metadata on the artifact
			artifact.SetMetaValue("output", "hello")
			return nil
		}),
	}
}

func (s *TestSetter) Read(p []byte) (n int, err error) {
	return s.buffer.Read(p)
}

func (s *TestSetter) Write(p []byte) (n int, err error) {
	return s.buffer.Write(p)
}

func (s *TestSetter) Close() error {
	return s.buffer.Close()
}

// ComplexSetter demonstrates setting multiple metadata values
type ComplexSetter struct {
	buffer *stream.Buffer
}

func NewComplexSetter() *ComplexSetter {
	return &ComplexSetter{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			artifact.SetMetaValue("name", "test")
			artifact.SetMetaValue("count", 42)
			artifact.SetMetaValue("processed", true)
			return nil
		}),
	}
}

func (s *ComplexSetter) Read(p []byte) (n int, err error) {
	return s.buffer.Read(p)
}

func (s *ComplexSetter) Write(p []byte) (n int, err error) {
	return s.buffer.Write(p)
}

func (s *ComplexSetter) Close() error {
	return s.buffer.Close()
}
