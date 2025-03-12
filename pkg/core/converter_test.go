package core

import (
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
			So(converter.Buffer, ShouldNotBeNil)
		})
	})
}
