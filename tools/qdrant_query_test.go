package tools

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestQdrantQuery(t *testing.T) {
	Convey("Given a new QdrantQuery instance", t, func() {
		collection := "test_collection"
		dimension := uint64(1536)
		query := NewQdrantQuery(collection, dimension)

		Convey("When checking the Name method", func() {
			name := query.Name()

			Convey("Then it should return 'qdrant'", func() {
				So(name, ShouldEqual, "qdrant")
			})
		})

		Convey("When checking the Description method", func() {
			desc := query.Description()

			Convey("Then it should return a non-empty description", func() {
				So(desc, ShouldEqual, "Interact with Qdrant")
			})
		})

		Convey("When generating the schema", func() {
			schema := query.GenerateSchema()

			Convey("Then it should return a valid schema", func() {
				So(schema, ShouldNotBeNil)
			})
		})

		Convey("When initializing the query", func() {
			err := query.Initialize()

			Convey("Then it should initialize successfully", func() {
				So(err, ShouldBeNil)
				So(query.client, ShouldNotBeNil)
				So(query.embedder, ShouldNotBeNil)
				So(query.ctx, ShouldNotBeNil)
			})
		})

		Convey("When connecting", func() {
			err := query.Connect(context.Background(), nil)

			Convey("Then it should connect successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When using the query", func() {
			args := map[string]any{
				"query": "test query",
			}
			result := query.Use(context.Background(), args)

			Convey("Then it should handle the query", func() {
				So(result, ShouldNotBeEmpty)
			})

			Convey("When query is missing from args", func() {
				result := query.Use(context.Background(), map[string]any{})

				Convey("Then it should return an error message", func() {
					So(result, ShouldEqual, "query is required")
				})
			})
		})
	})
}
