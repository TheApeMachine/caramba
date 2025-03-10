package core

import (
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewProcess tests the NewProcess constructor
func TestNewProcess(t *testing.T) {
	Convey("Given parameters for a new Process", t, func() {
		name := "test_process"
		description := "A test process"
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"id": map[string]string{
					"type": "string",
				},
				"value": map[string]string{
					"type": "number",
				},
			},
			"required": []string{"id", "value"},
		}

		Convey("When creating a new Process", func() {
			process := NewProcess(name, description, schema)

			Convey("Then the process should have the correct properties", func() {
				So(process, ShouldNotBeNil)
				So(process.Name, ShouldEqual, name)
				So(process.Description, ShouldEqual, description)
				So(process.Schema, ShouldResemble, schema)

				// Verify that buffers are initialized
				So(process.in, ShouldNotBeNil)
				So(process.out, ShouldNotBeNil)
				So(process.enc, ShouldNotBeNil)
				So(process.dec, ShouldNotBeNil)

				// Verify pre-encoding happened
				So(process.out.Len(), ShouldBeGreaterThan, 0)
			})
		})
	})
}

// TestProcessRead tests the Read method of Process
func TestProcessRead(t *testing.T) {
	Convey("Given a Process with a schema", t, func() {
		process := NewProcess(
			"test_process",
			"Test description",
			map[string]interface{}{"type": "object"},
		)
		buffer := make([]byte, 1024)

		Convey("When reading from the process", func() {
			n, err := process.Read(buffer)

			Convey("Then it should return process data as JSON", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				var parsed ProcessData
				err := json.Unmarshal(buffer[:n], &parsed)
				So(err, ShouldBeNil)
				So(parsed.Name, ShouldEqual, "test_process")
				So(parsed.Description, ShouldEqual, "Test description")

				// Check schema was included
				schemaMap, ok := parsed.Schema.(map[string]interface{})
				So(ok, ShouldBeTrue)
				So(schemaMap["type"], ShouldEqual, "object")
			})
		})

		Convey("When reading until empty", func() {
			// First read to consume buffer
			firstBuffer := make([]byte, 1024)
			process.Read(firstBuffer)

			// The process's Read method should re-encode if buffer is empty
			n, err := process.Read(buffer)

			Convey("Then it should still return data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)
			})
		})
	})
}

// TestProcessWrite tests the Write method of Process
func TestProcessWrite(t *testing.T) {
	Convey("Given a Process", t, func() {
		process := NewProcess(
			"original_name",
			"Original description",
			map[string]interface{}{"type": "string"},
		)

		Convey("When writing valid JSON data", func() {
			updatedData := &ProcessData{
				Name:        "updated_name",
				Description: "Updated description",
				Schema:      map[string]interface{}{"type": "number"},
			}

			jsonData, err := json.Marshal(updatedData)
			So(err, ShouldBeNil)

			n, err := process.Write(jsonData)

			Convey("Then it should update the process data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))
				So(process.Name, ShouldEqual, "updated_name")
				So(process.Description, ShouldEqual, "Updated description")

				schemaMap, ok := process.Schema.(map[string]interface{})
				So(ok, ShouldBeTrue)
				So(schemaMap["type"], ShouldEqual, "number")
			})

			Convey("And reading should return the updated data", func() {
				buffer := make([]byte, 1024)
				n, err := process.Read(buffer)

				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				var parsed ProcessData
				err = json.Unmarshal(buffer[:n], &parsed)
				So(err, ShouldBeNil)
				So(parsed.Name, ShouldEqual, "updated_name")
				So(parsed.Description, ShouldEqual, "Updated description")
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"name": "invalid" - broken json`)
			n, err := process.Write(invalidJSON)

			Convey("Then it should retain bytes but not update fields", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
				So(process.Name, ShouldEqual, "original_name")
				So(process.Description, ShouldEqual, "Original description")
			})
		})
	})
}

// TestProcessClose tests the Close method of Process
func TestProcessClose(t *testing.T) {
	Convey("Given a Process with data", t, func() {
		process := NewProcess(
			"test_process",
			"Test description",
			map[string]interface{}{"type": "object"},
		)

		Convey("When closing the process", func() {
			err := process.Close()

			Convey("Then it should reset all properties", func() {
				So(err, ShouldBeNil)
				So(process.Name, ShouldEqual, "")
				So(process.Description, ShouldEqual, "")
				So(process.Schema, ShouldBeNil)
			})
		})
	})
}
