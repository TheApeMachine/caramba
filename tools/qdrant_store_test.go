package tools

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestQdrantStore(t *testing.T) {
	Convey("Given a new QdrantStore instance", t, func() {
		collection := "test_collection"
		dimension := uint64(1536)
		store := NewQdrantStore(collection, dimension)

		Convey("When checking the Name method", func() {
			name := store.Name()

			Convey("Then it should return 'qdrant'", func() {
				So(name, ShouldEqual, "qdrant")
			})
		})

		Convey("When checking the Description method", func() {
			desc := store.Description()

			Convey("Then it should return a non-empty description", func() {
				So(desc, ShouldEqual, "Interact with Qdrant")
			})
		})

		Convey("When generating the schema", func() {
			schema := store.GenerateSchema()

			Convey("Then it should return a valid schema", func() {
				So(schema, ShouldNotBeNil)
			})
		})

		Convey("When initializing the store", func() {
			err := store.Initialize()

			Convey("Then it should initialize successfully", func() {
				So(err, ShouldBeNil)
				So(store.client, ShouldNotBeNil)
				So(store.embedder, ShouldNotBeNil)
				So(store.ctx, ShouldNotBeNil)
			})
		})

		Convey("When connecting", func() {
			err := store.Connect(context.Background(), nil)

			Convey("Then it should connect successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When using the store", func() {
			args := map[string]any{
				"documents": []interface{}{"test document"},
				"metadata": map[string]any{
					"test": "metadata",
				},
			}

			Convey("Then it should handle document storage", func() {
				result := store.Use(context.Background(), args)
				So(result, ShouldEqual, "memory saved in vector store")
			})

			Convey("When documents are missing from args", func() {
				result := store.Use(context.Background(), map[string]any{})
				So(result, ShouldEqual, "memory saved in vector store")
			})
		})
	})
}
