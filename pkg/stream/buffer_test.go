package stream

import (
	"io"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/event"
)

// testEvent creates a test event artifact with predefined data
func testEvent() *event.Artifact {
	return event.New(
		"test",
		event.MessageEvent,
		event.UserRole,
		[]byte("test-data"),
	)
}

// readEventData reads all data from an event artifact into a byte slice
func readEventData(evt *event.Artifact) ([]byte, int, error) {
	data := make([]byte, 1024)
	n, err := evt.Read(data)
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
func verifyEventData(evt *event.Artifact, expectedData []byte) {
	data, n, err := readEventData(evt)
	So(err, ShouldBeNil)
	So(n, ShouldEqual, len(expectedData))
	So(data, ShouldResemble, expectedData)
}

func TestNewBuffer(t *testing.T) {
	Convey("Given a new buffer", t, func() {
		var handlerCalled bool
		buffer := NewBuffer(func(event *event.Artifact) error {
			handlerCalled = true
			return nil
		})

		So(buffer, ShouldNotBeNil)
		So(buffer.event, ShouldNotBeNil)
		So(buffer.fn, ShouldNotBeNil)
		So(handlerCalled, ShouldBeFalse) // Verify initial state
	})
}

func TestRead(t *testing.T) {
	Convey("Given a new buffer", t, func() {
		buffer := NewBuffer(func(event *event.Artifact) error {
			return nil
		})

		Convey("When reading from an empty buffer without stream", func() {
			testData := testEvent()
			data, _, err := readEventData(testData)
			So(err, ShouldBeNil)

			buffer.event = testData
			verifyBufferRead(buffer, data, false)
		})

		Convey("When reading from a buffer with data in the stream", func() {
			testData := testEvent()
			expectedData, _, err := readEventData(testData)
			So(err, ShouldBeNil)

			// Read from the buffer
			verifyBufferRead(buffer, expectedData, true)
		})

		Convey("When reading from a closed stream", func() {
			p := make([]byte, 1024)
			n, err := buffer.Read(p)
			So(err, ShouldEqual, io.EOF)
			So(n, ShouldEqual, 0)
		})

		Convey("When reading from an empty stream", func() {
			p := make([]byte, 1024)
			n, err := buffer.Read(p)
			So(err, ShouldEqual, io.EOF)
			So(n, ShouldEqual, 0)
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given a new buffer", t, func() {
		var handlerCalled bool
		buffer := NewBuffer(func(event *event.Artifact) error {
			handlerCalled = true
			return nil
		})

		Convey("When writing valid data", func() {
			testData := testEvent()
			data, dataLen, err := readEventData(testData)
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
			buffer := NewBuffer(func(event *event.Artifact) error {
				return io.ErrUnexpectedEOF
			})

			testData := testEvent()
			data, dataLen, err := readEventData(testData)
			So(err, ShouldBeNil)

			n, err := buffer.Write(data)
			So(err, ShouldEqual, io.ErrUnexpectedEOF)
			So(n, ShouldEqual, dataLen)
		})

		Convey("When buffer event is nil", func() {
			buffer.event = nil // Use the existing buffer to ensure consistent setup
			n, err := buffer.Write([]byte("test"))
			So(err, ShouldBeError)
			So(n, ShouldEqual, 0)
		})
	})
}

func TestClose(t *testing.T) {
	Convey("Given a new buffer", t, func() {
		buffer := NewBuffer(func(event *event.Artifact) error {
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
		processed := make(chan *event.Artifact, 1)
		buffer := NewBuffer(func(evt *event.Artifact) error {
			processed <- evt
			return nil
		})

		Convey("When writing and reading through the buffer", func() {
			// Write test data
			testData := testEvent()
			data, dataLen, err := readEventData(testData)
			So(err, ShouldBeNil)

			n, err := buffer.Write(data)
			So(err, ShouldBeNil)
			So(n, ShouldEqual, dataLen)

			// Verify the processing function was called
			select {
			case processedEvent := <-processed:
				So(processedEvent, ShouldNotBeNil)
				verifyEventData(processedEvent, data)
			case <-time.After(time.Second):
				t.Fatal("Timeout waiting for event processing")
			}
		})
	})
}
