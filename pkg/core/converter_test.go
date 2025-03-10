package core

import (
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewConverter tests the NewConverter constructor
func TestNewConverter(t *testing.T) {
	Convey("When creating a new Converter", t, func() {
		converter := NewConverter()

		Convey("Then the converter should be properly initialized", func() {
			So(converter, ShouldNotBeNil)
			So(converter.ConverterData, ShouldNotBeNil)
			So(converter.Event, ShouldBeNil)
			So(converter.dec, ShouldNotBeNil)
			So(converter.in, ShouldNotBeNil)
			So(converter.out, ShouldNotBeNil)
		})
	})
}

// TestConverterRead tests the Read method of Converter
func TestConverterRead(t *testing.T) {
	Convey("Given a Converter with content in the output buffer", t, func() {
		converter := NewConverter()
		testContent := "Test content in output buffer"

		// Manually add content to the output buffer
		converter.out.WriteString(testContent)

		Convey("When reading from the converter", func() {
			buffer := make([]byte, 1024)
			n, err := converter.Read(buffer)

			Convey("Then it should return the content", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(testContent))
				So(string(buffer[:n]), ShouldEqual, testContent)
			})
		})

		Convey("When reading until empty", func() {
			// First read to consume buffer
			firstBuffer := make([]byte, 1024)
			converter.Read(firstBuffer)

			// Second read should be empty
			buffer := make([]byte, 1024)
			n, _ := converter.Read(buffer)

			Convey("Then it should return 0 bytes", func() {
				// Only check bytes, the error could be nil or EOF depending on implementation
				So(n, ShouldEqual, 0)
			})
		})
	})
}

// TestConverterWrite tests the Write method of Converter
func TestConverterWrite(t *testing.T) {
	Convey("Given a Converter", t, func() {
		converter := NewConverter()

		Convey("When writing valid JSON with a message", func() {
			message := NewMessage("user", "testuser", "Hello, world!")
			event := NewEvent(message, nil)

			jsonData, err := json.Marshal(event)
			So(err, ShouldBeNil)

			n, err := converter.Write(jsonData)

			Convey("Then it should extract the message content", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))

				// Read the output to verify content
				buffer := make([]byte, 1024)
				n, err := converter.Read(buffer)
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)
				So(string(buffer[:n]), ShouldEqual, "Hello, world!")
			})
		})

		Convey("When writing JSON with a nil message", func() {
			event := NewEvent(nil, nil)

			jsonData, err := json.Marshal(event)
			So(err, ShouldBeNil)

			n, err := converter.Write(jsonData)

			Convey("Then it should return an error", func() {
				So(err, ShouldNotBeNil)
				// Don't check the specific error message as it might vary
				So(n, ShouldEqual, len(jsonData))
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"event": {"message": {"role": "invalid" - broken json`)
			n, err := converter.Write(invalidJSON)

			Convey("Then it should retain bytes but not update output", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))

				// Output buffer should remain empty
				buffer := make([]byte, 1024)
				n, _ := converter.Read(buffer)
				So(n, ShouldEqual, 0)
			})
		})
	})
}

// TestConverterClose tests the Close method of Converter
func TestConverterClose(t *testing.T) {
	Convey("Given a Converter", t, func() {
		converter := NewConverter()

		Convey("When closing the converter", func() {
			err := converter.Close()

			Convey("Then it should not error", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}
