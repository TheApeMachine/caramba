package tools

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNeo4jStore(t *testing.T) {
	Convey("Given a new Neo4jStore instance", t, func() {
		store := NewNeo4jStore()

		Convey("When checking the Name method", func() {
			name := store.Name()

			Convey("Then it should return 'neo4j'", func() {
				So(name, ShouldEqual, "neo4j")
			})
		})

		Convey("When checking the Description method", func() {
			desc := store.Description()

			Convey("Then it should return a non-empty description", func() {
				So(desc, ShouldEqual, "Interact with Neo4j")
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
			})
		})

		Convey("When connecting", func() {
			err := store.Connect(context.Background(), nil)

			Convey("Then it should connect successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When using the store", func() {
			Convey("With valid cypher query", func() {
				args := map[string]any{
					"cypher": "CREATE (n:Test {name: 'test'}) RETURN n",
				}
				result := store.Use(context.Background(), args)

				Convey("Then it should handle the write operation", func() {
					So(result, ShouldEqual, "relationships stored in graph database")
				})
			})

			Convey("When cypher is missing from args", func() {
				result := store.Use(context.Background(), map[string]any{})

				Convey("Then it should return an error message", func() {
					So(result, ShouldEqual, "cypher is required")
				})
			})

			Convey("When write operation fails", func() {
				args := map[string]any{
					"cypher": "INVALID CYPHER QUERY",
				}
				result := store.Use(context.Background(), args)

				Convey("Then it should return error message", func() {
					So(result, ShouldNotEqual, "relationships stored in graph database")
				})
			})
		})
	})
}
