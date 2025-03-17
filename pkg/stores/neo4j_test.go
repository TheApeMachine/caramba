package stores

// import (
// 	"encoding/json"
// 	"testing"

// 	. "github.com/smartystreets/goconvey/convey"
// 	"github.com/theapemachine/caramba/pkg/core"
// )

// // TestNewNeo4j tests the NewNeo4j function
// func TestNewNeo4j(t *testing.T) {
// 	Convey("Given a collection name", t, func() {
// 		collection := "test_collection"

// 		Convey("When creating a new Neo4j store", func() {
// 			neo4j := NewNeo4j(collection)

// 			Convey("Then it should be properly initialized", func() {
// 				So(neo4j, ShouldNotBeNil)
// 				So(neo4j.Neo4jData, ShouldNotBeNil)
// 				So(neo4j.in, ShouldNotBeNil)
// 				So(neo4j.out, ShouldNotBeNil)
// 				So(neo4j.enc, ShouldNotBeNil)
// 				So(neo4j.dec, ShouldNotBeNil)
// 			})
// 		})
// 	})
// }

// // TestNeo4jWriteBasics tests basic operation of Neo4j.Write
// func TestNeo4jWriteBasics(t *testing.T) {
// 	Convey("Given a Neo4j store", t, func() {
// 		neo4j := NewNeo4j("test_collection")

// 		Convey("When writing valid JSON data", func() {
// 			event := core.NewEvent(
// 				core.NewMessage("user", "test", "test content"),
// 				nil,
// 			)

// 			eventBytes, _ := json.Marshal(event)
// 			n, err := neo4j.Write(eventBytes)

// 			Convey("Then it should accept the data without error", func() {
// 				So(err, ShouldBeNil)
// 				So(n, ShouldEqual, len(eventBytes))
// 			})
// 		})

// 		Convey("When writing invalid JSON", func() {
// 			invalidJSON := []byte(`{"broken": "json"`)
// 			n, err := neo4j.Write(invalidJSON)

// 			Convey("Then it should not fail", func() {
// 				So(err, ShouldBeNil)
// 				So(n, ShouldEqual, len(invalidJSON))
// 			})
// 		})
// 	})
// }

// // TestNeo4jClose tests the Close method
// func TestNeo4jClose(t *testing.T) {
// 	Convey("Given a Neo4j store", t, func() {
// 		neo4j := NewNeo4j("test_collection")

// 		Convey("When closing the store", func() {
// 			err := neo4j.Close()

// 			Convey("Then it should close successfully", func() {
// 				So(err, ShouldBeNil)
// 			})
// 		})
// 	})
// }
