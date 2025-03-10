package stores

import (
	"encoding/json"
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

		Convey("When writing valid JSON data", func() {
			event := core.NewEvent(
				core.NewMessage("user", "test", "test content"),
				nil,
			)

			eventBytes, _ := json.Marshal(event)
			n, err := qdrant.Write(eventBytes)

			Convey("Then it should accept the data without error", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(eventBytes))
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"broken": "json"`)
			n, err := qdrant.Write(invalidJSON)

			Convey("Then it should not fail", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
			})
		})
	})
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
