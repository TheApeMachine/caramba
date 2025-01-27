package tools

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNeo4jQuery(t *testing.T) {
	Convey("Given a new Neo4jQuery instance", t, func() {
		query := NewNeo4jQuery()

		Convey("When checking the Name method", func() {
			name := query.Name()

			Convey("Then it should return 'neo4j'", func() {
				So(name, ShouldEqual, "neo4j")
			})
		})

		Convey("When checking the Description method", func() {
			desc := query.Description()

			Convey("Then it should return a non-empty description", func() {
				So(desc, ShouldEqual, "Interact with Neo4j")
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
			})
		})

		Convey("When connecting", func() {
			err := query.Connect(context.Background(), nil)

			Convey("Then it should connect successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When using the query", func() {
			Convey("With valid cypher query", func() {
				args := map[string]any{
					"cypher": "MATCH (n) RETURN n LIMIT 1",
				}
				result := query.Use(context.Background(), args)

				Convey("Then it should handle the query", func() {
					So(result, ShouldNotBeEmpty)
				})
			})

			Convey("When cypher is missing from args", func() {
				result := query.Use(context.Background(), map[string]any{})

				Convey("Then it should return an error message", func() {
					So(result, ShouldEqual, "cypher is required")
				})
			})

			Convey("When query returns no results", func() {
				args := map[string]any{
					"cypher": "MATCH (n:NonExistent) RETURN n LIMIT 1",
				}
				result := query.Use(context.Background(), args)

				Convey("Then it should return no results message", func() {
					So(result, ShouldEqual, "No results found")
				})
			})
		})
	})
}
