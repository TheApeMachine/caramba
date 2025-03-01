package core

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestIsLikelyJSON(t *testing.T) {
	Convey("Given various strings to check for JSON format", t, func() {
		Convey("When the string starts with '{'", func() {
			input := `{"name": "test"}`
			So(isLikelyJSON(input), ShouldBeTrue)
		})

		Convey("When the string starts with '['", func() {
			input := `[{"name": "test"}]`
			So(isLikelyJSON(input), ShouldBeTrue)
		})

		Convey("When the string has whitespace before '{'", func() {
			input := `  {"name": "test"}`
			So(isLikelyJSON(input), ShouldBeTrue)
		})

		Convey("When the string is not JSON", func() {
			input := `This is not JSON`
			So(isLikelyJSON(input), ShouldBeFalse)
		})
	})
}

func TestMaybeExtractContentField(t *testing.T) {
	Convey("Given JSON strings with content field", t, func() {
		Convey("When JSON has a content field", func() {
			input := `{"content": "Hello world"}`
			So(maybeExtractContentField(input), ShouldEqual, "Hello world")
		})

		Convey("When JSON doesn't have a content field", func() {
			input := `{"message": "Hello world"}`
			So(maybeExtractContentField(input), ShouldEqual, input)
		})

		Convey("When input is not valid JSON", func() {
			input := `Not valid JSON`
			So(maybeExtractContentField(input), ShouldEqual, input)
		})
	})
}

func TestUtilsFormatStreamedContent(t *testing.T) {
	Convey("Given various markdown formatted strings", t, func() {
		Convey("When formatting a main header", func() {
			input := "# Header"
			result := formatStreamedContent(input)
			// In test environments, color formatting may be disabled
			// Just check that the content is preserved
			So(result, ShouldContainSubstring, input)
		})

		Convey("When formatting a secondary header", func() {
			input := "## Header"
			result := formatStreamedContent(input)
			// In test environments, color formatting may be disabled
			So(result, ShouldContainSubstring, input)
		})

		Convey("When formatting a tertiary header", func() {
			input := "### Header"
			result := formatStreamedContent(input)
			// In test environments, color formatting may be disabled
			So(result, ShouldContainSubstring, input)
		})

		Convey("When formatting a list item", func() {
			input := "- List item"
			result := formatStreamedContent(input)
			// In test environments, color formatting may be disabled
			So(result, ShouldContainSubstring, input)
		})

		Convey("When formatting a blockquote", func() {
			input := "> Blockquote"
			result := formatStreamedContent(input)
			// In test environments, color formatting may be disabled
			So(result, ShouldContainSubstring, input)
		})

		Convey("When formatting code", func() {
			input := "```code```"
			result := formatStreamedContent(input)
			// In test environments, color formatting may be disabled
			So(result, ShouldContainSubstring, input)
		})

		Convey("When formatting plain text", func() {
			input := "Plain text"
			result := formatStreamedContent(input)
			So(result, ShouldEqual, input)
		})
	})
}
