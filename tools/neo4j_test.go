package tools

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNeo4j(t *testing.T) {
	Convey("Given a new Neo4j instance", t, func() {
		neo4j, err := NewNeo4j()
		So(err, ShouldNotBeNil)   // We expect an error since we can't connect to Neo4j
		So(neo4j, ShouldNotBeNil) // But we should still get a Neo4j instance

		Convey("When initializing without a connection", func() {
			err := neo4j.Initialize()

			Convey("Then it should return an error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "not initialized")
			})
		})

		Convey("When connecting without a connection", func() {
			err := neo4j.Connect()

			Convey("Then it should return an error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "not initialized")
			})
		})

		Convey("When using the tool without a connection", func() {
			ctx := context.Background()

			Convey("With query operation", func() {
				neo4j.Operation = "query"
				args := map[string]any{
					"cypher": "MATCH (n) RETURN n LIMIT 1",
				}
				result := neo4j.Use(ctx, args)

				Convey("Then it should return an error message", func() {
					So(result, ShouldContainSubstring, "not initialized")
				})
			})

			Convey("With write operation", func() {
				neo4j.Operation = "write"
				args := map[string]any{
					"query": "CREATE (n:Test {name: 'test'}) RETURN n",
				}
				result := neo4j.Use(ctx, args)

				Convey("Then it should return an error message", func() {
					So(result, ShouldContainSubstring, "not initialized")
				})
			})

			Convey("With invalid operation", func() {
				neo4j.Operation = "invalid"
				args := map[string]any{}
				result := neo4j.Use(ctx, args)

				Convey("Then it should return error message", func() {
					So(result, ShouldEqual, "Unsupported operation")
				})
			})
		})

		Convey("When querying without a connection", func() {
			query := "MATCH (n) RETURN n LIMIT 1"
			results, err := neo4j.Query(query)

			Convey("Then it should return an error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "not initialized")
				So(results, ShouldBeNil)
			})
		})

		Convey("When writing without a connection", func() {
			query := "CREATE (n:Test {name: 'test'}) RETURN n"
			err := neo4j.Write(query)

			Convey("Then it should return an error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "not initialized")
			})
		})

		Convey("When closing without a connection", func() {
			err := neo4j.Close()

			Convey("Then it should return an error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "not initialized")
			})
		})
	})
}
