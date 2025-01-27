package tools

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNeo4j(t *testing.T) {
	Convey("Given a new Neo4j instance", t, func() {
		neo4j := NewNeo4j()

		Convey("When creating a new instance", func() {
			Convey("Then it should be properly initialized", func() {
				So(neo4j, ShouldNotBeNil)
				So(neo4j.client, ShouldNotBeNil)
			})
		})

		Convey("When initializing", func() {
			err := neo4j.Initialize()

			Convey("Then it should initialize successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When connecting", func() {
			err := neo4j.Connect()

			Convey("Then it should connect successfully", func() {
				So(err, ShouldBeNil)
			})
		})

		Convey("When using the tool", func() {
			ctx := context.Background()

			Convey("With query operation", func() {
				neo4j.Operation = "query"
				args := map[string]any{
					"cypher": "MATCH (n) RETURN n LIMIT 1",
				}
				result := neo4j.Use(ctx, args)

				Convey("Then it should handle query operation", func() {
					So(result, ShouldNotBeEmpty)
				})
			})

			Convey("With write operation", func() {
				neo4j.Operation = "write"
				args := map[string]any{
					"query": "CREATE (n:Test {name: 'test'}) RETURN n",
				}
				result := neo4j.Use(ctx, args)

				Convey("Then it should handle write operation", func() {
					So(result, ShouldNotBeEmpty)
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

		Convey("When querying", func() {
			query := "MATCH (n) RETURN n LIMIT 1"
			results, err := neo4j.Query(query)

			Convey("Then it should perform query successfully", func() {
				So(err, ShouldBeNil)
				So(results, ShouldNotBeNil)
			})
		})

		Convey("When writing", func() {
			query := "CREATE (n:Test {name: 'test'}) RETURN n"
			result := neo4j.Write(query)

			Convey("Then it should perform write successfully", func() {
				So(result, ShouldNotBeNil)
				So(result.Err(), ShouldBeNil)
			})
		})

		Convey("When closing", func() {
			err := neo4j.Close()

			Convey("Then it should close successfully", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}
