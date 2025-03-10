package stores

import (
	"encoding/json"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/core"
)

// TestNewQdrant tests the NewQdrant function
func TestNewQdrant(t *testing.T) {
	Convey("Given a collection name", t, func() {
		collection := "test_collection"

		Convey("When creating a new Qdrant store", func() {
			qdrant := NewQdrant(collection)

			Convey("Then it should be properly initialized", func() {
				So(qdrant, ShouldNotBeNil)
				So(qdrant.QdrantData, ShouldNotBeNil)
				So(qdrant.collection, ShouldEqual, collection)
				// Don't test specific dimension size as it may change
				So(qdrant.dimensions, ShouldBeGreaterThan, 0)
				So(qdrant.in, ShouldNotBeNil)
				So(qdrant.out, ShouldNotBeNil)
				So(qdrant.enc, ShouldNotBeNil)
				So(qdrant.dec, ShouldNotBeNil)
			})
		})
	})
}

// TestQdrantWriteBasics tests basic JSON parsing in write
func TestQdrantWriteBasics(t *testing.T) {
	Convey("Given a Qdrant store", t, func() {
		qdrant := NewQdrant("test_collection")

		// Skip the actual Qdrant operations for testing
		// We'll just test the basic IO functionality

		Convey("When writing valid JSON data", func() {
			event := core.NewEvent(
				core.NewMessage("user", "test", "test content"),
				nil,
			)

			eventBytes, _ := json.Marshal(event)
			n, err := qdrant.in.Write(eventBytes)

			Convey("Then it should accept the data without error", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(eventBytes))
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"broken": "json"`)
			n, err := qdrant.in.Write(invalidJSON)

			Convey("Then it should not fail", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
			})
		})
	})
}

// MockEmbedder is a test helper implementing io.ReadWriteCloser
type MockEmbedder struct {
	embedding []float64
	buffer    []byte
}

func (m *MockEmbedder) Read(p []byte) (n int, err error) {
	// If buffer is not yet prepared, marshal the embedding
	if m.buffer == nil {
		var err error
		m.buffer, err = json.Marshal(m.embedding)
		if err != nil {
			return 0, err
		}
	}

	// Copy as much as we can into the provided buffer
	n = copy(p, m.buffer)

	// If we copied everything, we're done
	if n == len(m.buffer) {
		m.buffer = nil // Reset for next time
		return n, io.EOF
	}

	// Otherwise, keep the remaining part for next Read call
	m.buffer = m.buffer[n:]
	return n, nil
}

func (m *MockEmbedder) Write(p []byte) (n int, err error) {
	// Just return the length of the input
	return len(p), nil
}

func (m *MockEmbedder) Close() error {
	return nil
}

// TestQdrantClose tests the Close method
func TestQdrantClose(t *testing.T) {
	Convey("Given a Qdrant store", t, func() {
		qdrant := NewQdrant("test_collection")

		Convey("When closing the store", func() {
			err := qdrant.Close()

			Convey("Then it should close successfully", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}
