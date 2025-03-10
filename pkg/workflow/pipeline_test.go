package workflow

import (
	"errors"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// MockReadWriteCloser is a test helper implementing io.ReadWriteCloser
type MockReadWriteCloser struct {
	readFunc  func(p []byte) (n int, err error)
	writeFunc func(p []byte) (n int, err error)
	closeFunc func() error
}

func (m *MockReadWriteCloser) Read(p []byte) (n int, err error) {
	if m.readFunc != nil {
		return m.readFunc(p)
	}
	// Default behavior: return EOF immediately
	return 0, io.EOF
}

func (m *MockReadWriteCloser) Write(p []byte) (n int, err error) {
	if m.writeFunc != nil {
		return m.writeFunc(p)
	}
	return len(p), nil
}

func (m *MockReadWriteCloser) Close() error {
	if m.closeFunc != nil {
		return m.closeFunc()
	}
	return nil
}

// TestNewPipeline tests the NewPipeline constructor
func TestNewPipeline(t *testing.T) {
	Convey("Given a set of components", t, func() {
		comp1 := &MockReadWriteCloser{}
		comp2 := &MockReadWriteCloser{}
		comp3 := &MockReadWriteCloser{}

		Convey("When creating a new Pipeline with multiple components", func() {
			pipeline := NewPipeline(comp1, comp2, comp3)

			Convey("Then the pipeline should be created successfully", func() {
				So(pipeline, ShouldNotBeNil)
				So(pipeline, ShouldHaveSameTypeAs, &Pipeline{})

				// Cast to access internal components
				p := pipeline.(*Pipeline)
				So(p.components, ShouldHaveLength, 3)
				So(p.components[0], ShouldEqual, comp1)
				So(p.components[1], ShouldEqual, comp2)
				So(p.components[2], ShouldEqual, comp3)
			})
		})

		Convey("When creating a pipeline with just one component", func() {
			pipeline := NewPipeline(comp1)

			Convey("Then it should still create the pipeline", func() {
				So(pipeline, ShouldNotBeNil)

				p := pipeline.(*Pipeline)
				So(p.components, ShouldHaveLength, 1)
			})
		})
	})
}

// TestPipelineComponents tests that pipeline properly connects its components
func TestPipelineComponents(t *testing.T) {
	Convey("Given a pipeline with components", t, func() {
		// Simple mocks that verify a write to first component returns data from last component
		firstComponentWritten := false
		finalComponentRead := false

		// First component: record that it was written to, and always return EOF on read
		comp1 := &MockReadWriteCloser{
			readFunc: func(p []byte) (int, error) {
				// Always return EOF to end the copy operation immediately
				return 0, io.EOF
			},
			writeFunc: func(p []byte) (int, error) {
				firstComponentWritten = true
				return len(p), nil
			},
		}

		// Second component: record that it was read from
		comp2 := &MockReadWriteCloser{
			readFunc: func(p []byte) (int, error) {
				finalComponentRead = true
				// Always return EOF
				return 0, io.EOF
			},
		}

		pipeline := NewPipeline(comp1, comp2)

		Convey("When using the pipeline", func() {
			// Just test that writing data flows through
			pipeline.Write([]byte("test"))

			// And reading should hit the last component
			buf := make([]byte, 10)
			pipeline.Read(buf)

			Convey("Then it should use all components", func() {
				So(firstComponentWritten, ShouldBeTrue)
				So(finalComponentRead, ShouldBeTrue)
			})
		})
	})
}

// TestPipelineWrite tests the Write method of Pipeline
func TestPipelineWrite(t *testing.T) {
	Convey("Given a pipeline with components", t, func() {
		firstWriteCalled := false
		comp1 := &MockReadWriteCloser{
			writeFunc: func(p []byte) (int, error) {
				firstWriteCalled = true
				return len(p), nil
			},
		}

		comp2 := &MockReadWriteCloser{}
		comp3 := &MockReadWriteCloser{}

		pipeline := NewPipeline(comp1, comp2, comp3)

		Convey("When writing to the pipeline", func() {
			data := []byte("test data")
			n, err := pipeline.Write(data)

			Convey("Then it should write to the first component only", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(data))
				So(firstWriteCalled, ShouldBeTrue)
			})
		})

		Convey("When first component write fails", func() {
			expectedErr := errors.New("write error")
			failingComp := &MockReadWriteCloser{
				writeFunc: func(p []byte) (int, error) {
					return 3, expectedErr // Partial write with error
				},
			}

			failingPipeline := NewPipeline(failingComp, comp2)
			data := []byte("test data")
			n, err := failingPipeline.Write(data)

			Convey("Then it should return the error and bytes written", func() {
				So(err, ShouldEqual, expectedErr)
				So(n, ShouldEqual, 3)
			})
		})
	})
}

// TestPipelineClose tests the Close method of Pipeline
func TestPipelineClose(t *testing.T) {
	Convey("Given a pipeline with components", t, func() {
		closeCount := 0

		createMockWithClose := func() *MockReadWriteCloser {
			return &MockReadWriteCloser{
				closeFunc: func() error {
					closeCount++
					return nil
				},
			}
		}

		comp1 := createMockWithClose()
		comp2 := createMockWithClose()
		comp3 := createMockWithClose()

		pipeline := NewPipeline(comp1, comp2, comp3)

		Convey("When closing the pipeline", func() {
			err := pipeline.Close()

			Convey("Then it should close all components", func() {
				So(err, ShouldBeNil)
				So(closeCount, ShouldEqual, 3)
			})
		})

		Convey("When a component fails to close", func() {
			expectedErr := errors.New("close error")
			failingComp := &MockReadWriteCloser{
				closeFunc: func() error {
					return expectedErr
				},
			}

			// Reset counter
			closeCount = 0

			// One component will fail, but all should be closed
			mixedPipeline := NewPipeline(comp1, failingComp, comp3)
			err := mixedPipeline.Close()

			Convey("Then it should still close all components and return first error", func() {
				So(err, ShouldEqual, expectedErr)
				So(closeCount, ShouldEqual, 2) // The non-failing components
			})
		})
	})
}
