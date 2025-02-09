package utils

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestJoinWith(t *testing.T) {
	Convey("Given JoinWith function", t, func() {
		Convey("Should join strings with delimiter", func() {
			result := JoinWith("-", "a", "b", "c")
			So(result, ShouldEqual, "a-b-c")
		})

		Convey("Should handle empty strings", func() {
			result := JoinWith(",", "", "b", "")
			So(result, ShouldEqual, ",b,")
		})

		Convey("Should handle single string", func() {
			result := JoinWith("-", "alone")
			So(result, ShouldEqual, "alone")
		})
	})
}

func TestReplaceWith(t *testing.T) {
	Convey("Given ReplaceWith function", t, func() {
		Convey("Should replace multiple placeholders", func() {
			template := "Hello {name}, your age is {age}"
			replacements := [][]string{
				{"name", "John"},
				{"age", "30"},
			}
			result := ReplaceWith(template, replacements)
			So(result, ShouldEqual, "Hello John, your age is 30")
		})

		Convey("Should handle no replacements", func() {
			template := "Hello {name}"
			result := ReplaceWith(template, [][]string{})
			So(result, ShouldEqual, "Hello {name}")
		})

		Convey("Should handle multiple occurrences of same placeholder", func() {
			template := "Hello {name}, bye {name}"
			replacements := [][]string{{"name", "John"}}
			result := ReplaceWith(template, replacements)
			So(result, ShouldEqual, "Hello John, bye John")
		})
	})
}

func TestNewID(t *testing.T) {
	Convey("Given NewID function", t, func() {
		Convey("Should generate unique IDs", func() {
			id1 := NewID()
			id2 := NewID()
			So(id1, ShouldNotEqual, id2)
			So(len(id1), ShouldEqual, 36) // UUID length
		})
	})
}

func TestNewName(t *testing.T) {
	Convey("Given NewName function", t, func() {
		Convey("Should generate unique names", func() {
			name1 := NewName()
			name2 := NewName()
			So(name1, ShouldNotEqual, "")
			So(name2, ShouldNotEqual, "")
			So(name1, ShouldNotEqual, name2)
		})
	})
}

func TestExtractJSONBlocks(t *testing.T) {
	Convey("Given ExtractJSONBlocks function", t, func() {
		Convey("Should extract valid JSON blocks", func() {
			input := "Some text\n```json\n{\"key\": \"value\"}\n```\nMore text"
			result := ExtractJSONBlocks(input)
			So(len(result), ShouldEqual, 1)
			So(result[0]["key"], ShouldEqual, "value")
		})

		Convey("Should handle multiple JSON blocks", func() {
			input := "```json\n{\"a\": 1}\n```\n```json\n{\"b\": 2}\n```"
			result := ExtractJSONBlocks(input)
			So(len(result), ShouldEqual, 2)
			So(result[0]["a"], ShouldEqual, 1.0)
			So(result[1]["b"], ShouldEqual, 2.0)
		})

		Convey("Should ignore invalid JSON", func() {
			input := "```json\n{invalid json}\n```"
			result := ExtractJSONBlocks(input)
			So(len(result), ShouldEqual, 0)
		})
	})
}

func TestQuickWrap(t *testing.T) {
	Convey("Given QuickWrap function", t, func() {
		Convey("Should wrap content in XML-like tags", func() {
			result := QuickWrap("test", "content", 0)
			So(result, ShouldEqual, "[test]\n\tcontent\n[/test]")
		})

		Convey("Should handle empty content", func() {
			result := QuickWrap("test", "", 0)
			So(result, ShouldEqual, "[test]\n\t\n[/test]")
		})
	})
}
