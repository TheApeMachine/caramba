package stream

import (
	"io"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/datura"
)

// testEvent creates a test event artifact with predefined data
func testArtifact() *datura.Artifact {
	return datura.New(
		datura.WithPayload([]byte("test-data")),
	)
}

// readEventData reads all data from an event artifact into a byte slice
func readArtifactData(artifact *datura.Artifact) ([]byte, int, error) {
	data := make([]byte, 1024)
	n, err := artifact.Read(data)
	if err == io.EOF {
		return data[:n], n, nil
	}
	return data[:n], n, err
}

// verifyBufferRead reads from the buffer and verifies the result
func verifyBufferRead(buffer *Buffer, expectedData []byte, streaming bool) {
	p := make([]byte, len(expectedData))
	n, err := buffer.Read(p)

	if streaming {
		So(err, ShouldBeNil)
	} else {
		So(err, ShouldEqual, io.EOF)
	}

	So(n, ShouldEqual, len(expectedData))
	So(p, ShouldResemble, expectedData)
}

// verifyEventData verifies that the event data matches the expected data
func verifyArtifactData(artifact *datura.Artifact, expectedData []byte) {
	data, n, err := readArtifactData(artifact)
	So(err, ShouldBeNil)
	So(n, ShouldEqual, len(expectedData))
	So(data, ShouldResemble, expectedData)
}

func TestNewBuffer(t *testing.T) {
	Convey("Given a new buffer", t, func() {
		var handlerCalled bool
		buffer := NewBuffer(func(artifact *datura.Artifact) error {
			handlerCalled = true
			return nil
		})

		So(buffer, ShouldNotBeNil)
		So(buffer.artifact, ShouldNotBeNil)
		So(buffer.fn, ShouldNotBeNil)
		So(handlerCalled, ShouldBeFalse) // Verify initial state
	})
}

func TestRead(t *testing.T) {
	Convey("Given a new buffer", t, func() {
		buffer := NewBuffer(func(artifact *datura.Artifact) error {
			return nil
		})

		Convey("When reading from a nil artifact", func() {
			buffer.artifact = nil
			p := make([]byte, 1024)
			n, err := buffer.Read(p)
			So(err, ShouldEqual, io.EOF)
			So(n, ShouldEqual, 0)
		})

		Convey("When reading after writing data", func() {
			testData := testArtifact()
			data, dataLen, err := readArtifactData(testData)
			So(err, ShouldBeNil)

			// Write data to the buffer first
			n, err := buffer.Write(data)
			So(err, ShouldBeNil)
			So(n, ShouldEqual, dataLen)

			// Read the data back
			p := make([]byte, len(data))
			n, err = buffer.Read(p)
			So(err, ShouldEqual, io.EOF) // Buffer.Read always returns EOF after reading
			So(n, ShouldEqual, len(data))
			So(p, ShouldResemble, data)
		})

		Convey("When reading with a small buffer", func() {
			testData := testArtifact()
			data, dataLen, err := readArtifactData(testData)
			So(err, ShouldBeNil)

			// Write data to the buffer
			n, err := buffer.Write(data)
			So(err, ShouldBeNil)
			So(n, ShouldEqual, dataLen)

			// Read with a smaller buffer
			p := make([]byte, 1) // Intentionally small buffer
			n, err = buffer.Read(p)
			So(err, ShouldEqual, io.ErrShortBuffer)
			So(n, ShouldBeGreaterThan, 0)
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given a new buffer", t, func() {
		var handlerCalled bool
		buffer := NewBuffer(func(artifact *datura.Artifact) error {
			handlerCalled = true
			return nil
		})

		Convey("When writing valid data", func() {
			testData := testArtifact()
			data, dataLen, err := readArtifactData(testData)
			So(err, ShouldBeNil)

			n, err := buffer.Write(data)
			So(err, ShouldBeNil)
			So(n, ShouldEqual, dataLen)
			So(handlerCalled, ShouldBeTrue)
		})

		Convey("When writing empty data", func() {
			n, err := buffer.Write([]byte{})
			So(err, ShouldBeError)
			So(err.Error(), ShouldEqual, "empty input")
			So(n, ShouldEqual, 0)
		})

		Convey("When handler returns an error", func() {
			buffer := NewBuffer(func(artifact *datura.Artifact) error {
				return io.ErrUnexpectedEOF
			})

			testData := testArtifact()
			data, dataLen, err := readArtifactData(testData)
			So(err, ShouldBeNil)

			n, err := buffer.Write(data)
			So(err, ShouldEqual, io.ErrUnexpectedEOF)
			So(n, ShouldEqual, dataLen)
		})

		Convey("When buffer event is nil", func() {
			buffer.artifact = nil // Use the existing buffer to ensure consistent setup
			n, err := buffer.Write([]byte("test"))
			So(err, ShouldBeError)
			So(n, ShouldEqual, 0)
		})
	})
}

func TestClose(t *testing.T) {
	Convey("Given a new buffer", t, func() {
		buffer := NewBuffer(func(artifact *datura.Artifact) error {
			return nil
		})

		Convey("When closing the buffer", func() {
			err := buffer.Close()
			So(err, ShouldBeNil)
		})
	})
}

func TestBufferIntegration(t *testing.T) {
	Convey("Given a buffer with a processing function", t, func() {
		processed := make(chan *datura.Artifact, 1)
		buffer := NewBuffer(func(artifact *datura.Artifact) error {
			processed <- artifact
			return nil
		})

		Convey("When writing and reading through the buffer", func() {
			// Write test data
			testData := testArtifact()
			data, dataLen, err := readArtifactData(testData)
			So(err, ShouldBeNil)

			n, err := buffer.Write(data)
			So(err, ShouldBeNil)
			So(n, ShouldEqual, dataLen)

			// Verify the processing function was called
			select {
			case processedArtifact := <-processed:
				So(processedArtifact, ShouldNotBeNil)
				verifyArtifactData(processedArtifact, data)
			case <-time.After(time.Second):
				t.Fatal("Timeout waiting for event processing")
			}
		})
	})
}
