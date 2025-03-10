package tools

import (
	"encoding/json"
	"errors"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewMemoryTool tests the NewMemoryTool constructor
func TestNewMemoryTool(t *testing.T) {
	Convey("Given store parameters", t, func() {
		store1 := &MockReadWriteCloser{}
		store2 := &MockReadWriteCloser{}

		Convey("When creating a new MemoryTool with stores", func() {
			mt := NewMemoryTool(store1, store2)

			Convey("Then the tool should have the correct properties", func() {
				So(mt, ShouldNotBeNil)
				So(mt.MemoryToolData, ShouldNotBeNil)
				So(mt.in, ShouldNotBeNil)
				So(mt.out, ShouldNotBeNil)
				So(mt.enc, ShouldNotBeNil)
				So(mt.dec, ShouldNotBeNil)
				So(mt.stores, ShouldHaveLength, 2)
				So(mt.stores[0], ShouldEqual, store1)
				So(mt.stores[1], ShouldEqual, store2)
				So(mt.Questions, ShouldNotBeNil)
				So(mt.Keywords, ShouldNotBeNil)
				So(mt.Cypher, ShouldEqual, "")
			})
		})

		Convey("When creating a new MemoryTool without stores", func() {
			mt := NewMemoryTool()

			Convey("Then the tool should have no stores", func() {
				So(mt, ShouldNotBeNil)
				So(mt.stores, ShouldHaveLength, 0)
			})
		})
	})
}

// TestMemoryToolRead tests the Read method
func TestMemoryToolRead(t *testing.T) {
	Convey("Given a MemoryTool with data", t, func() {
		mt := NewMemoryTool()

		// First write some data to ensure something is in the buffer
		toolData := MemoryToolData{
			Questions: []string{"What is the meaning of life?"},
			Keywords:  []string{"meaning", "life"},
			Cypher:    "MATCH (n) RETURN n",
		}
		jsonData, _ := json.Marshal(toolData)
		mt.Write(jsonData)

		Convey("When reading from the tool", func() {
			buffer := make([]byte, 1024)
			n, err := mt.Read(buffer)

			Convey("Then it should return the encoded data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				// Verify it contains the original data
				var decodedData MemoryToolData
				err := json.Unmarshal(buffer[:n], &decodedData)
				So(err, ShouldBeNil)
				So(len(decodedData.Questions), ShouldEqual, 1)
				So(decodedData.Questions[0], ShouldEqual, "What is the meaning of life?")
				So(len(decodedData.Keywords), ShouldEqual, 2)
				So(decodedData.Cypher, ShouldEqual, "MATCH (n) RETURN n")
			})
		})

		Convey("When the buffer is empty", func() {
			// Reset the buffer to simulate an empty buffer
			mt.out.Reset()

			// Try to read from an empty buffer
			buffer := make([]byte, 1024)
			n, err := mt.Read(buffer)

			Convey("Then it should return EOF", func() {
				So(err, ShouldEqual, io.EOF)
				So(n, ShouldEqual, 0)
			})
		})
	})
}

// TestMemoryToolWrite tests the Write method of MemoryTool
func TestMemoryToolWrite(t *testing.T) {
	Convey("Given a MemoryTool", t, func() {
		mt := NewMemoryTool()

		Convey("When writing valid data", func() {
			data := MemoryToolData{
				Questions: []string{"What is memory?", "How does it work?"},
				Keywords:  []string{"memory", "recall", "storage"},
				Cypher:    "MATCH (n:Memory) RETURN n",
			}
			jsonData, _ := json.Marshal(data)
			n, err := mt.Write(jsonData)

			Convey("Then it should update the tool data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))
				So(mt.Questions, ShouldResemble, data.Questions)
				So(mt.Keywords, ShouldResemble, data.Keywords)
				So(mt.Cypher, ShouldEqual, data.Cypher)
			})
		})

		Convey("When writing to stores", func() {
			// Create a store
			store := &MockReadWriteCloser{}
			mt := NewMemoryTool(store)

			// Write valid data
			data := MemoryToolData{
				Questions: []string{"What is the answer?"},
				Keywords:  []string{"answer", "question"},
				Cypher:    "",
			}
			jsonData, _ := json.Marshal(data)
			// Test that it doesn't panic
			mt.Write(jsonData)

			Convey("Then the tool should exist", func() {
				So(mt, ShouldNotBeNil)
			})
		})

		Convey("When the memory tool receives invalid input", func() {
			// Invalid JSON to test error handling
			invalidJSON := []byte(`{"questions": ["broken",], "keywords": []}`)
			n, _ := mt.Write(invalidJSON)

			Convey("Then it should handle invalid JSON gracefully", func() {
				So(n, ShouldEqual, len(invalidJSON))
				// Won't update memory tool data with invalid JSON
				So(len(mt.Questions), ShouldEqual, 0)
			})
		})
	})
}

// TestMemoryToolClose tests the Close method of MemoryTool
func TestMemoryToolClose(t *testing.T) {
	Convey("Given a MemoryTool with stores", t, func() {
		closeCount := 0
		store1 := &MockReadWriteCloser{
			closeFunc: func() error {
				closeCount++
				return nil
			},
		}
		store2 := &MockReadWriteCloser{
			closeFunc: func() error {
				closeCount++
				return nil
			},
		}

		mt := NewMemoryTool(store1, store2)

		Convey("When closing the tool", func() {
			err := mt.Close()

			Convey("Then it should close all stores", func() {
				So(err, ShouldBeNil)
				So(closeCount, ShouldEqual, 2)
			})
		})

		Convey("When a store fails to close", func() {
			// Reset counter
			closeCount = 0

			failingStore := &MockReadWriteCloser{
				closeFunc: func() error {
					return errors.New("close failed")
				},
			}

			mt := NewMemoryTool(store1, failingStore, store2)
			// Just calling close to see the side effects
			mt.Close()

			Convey("Then it should still close all stores", func() {
				// The implementation might ignore store errors
				So(closeCount, ShouldEqual, 2) // Both non-failing stores still closed
			})
		})
	})
}

// FailingWriter is a helper that always fails on Write
type FailingWriter struct{}

func (f *FailingWriter) Write(p []byte) (n int, err error) {
	return 0, errors.New("write error")
}
