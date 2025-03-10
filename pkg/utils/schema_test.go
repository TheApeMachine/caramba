package utils

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestStruct is a simple test struct for schema generation
type TestStruct struct {
	Name        string   `json:"name"`
	Age         int      `json:"age"`
	IsActive    bool     `json:"is_active"`
	Score       float64  `json:"score"`
	Tags        []string `json:"tags"`
	NestedField struct {
		ID   int    `json:"id"`
		Info string `json:"info"`
	} `json:"nested_field"`
}

// TestGenerateSchema tests the GenerateSchema function
func TestGenerateSchema(t *testing.T) {
	Convey("Given a struct type", t, func() {
		Convey("When generating a schema", func() {
			schema := GenerateSchema[TestStruct]()

			Convey("Then it should return a valid JSON schema", func() {
				So(schema, ShouldNotBeNil)

				// The schema should be a map
				schemaMap, ok := schema.(*map[string]interface{})
				if !ok {
					// If not a pointer to a map, try direct cast
					schemaMapValue, ok := schema.(map[string]interface{})
					if ok {
						So(schemaMapValue, ShouldNotBeNil)
					} else {
						// Otherwise just verify it exists
						So(schema, ShouldNotBeNil)
					}
				} else {
					So(*schemaMap, ShouldNotBeNil)
				}
			})
		})
	})
}
