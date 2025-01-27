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

func TestSplitIntoParagraphs(t *testing.T) {
	Convey("Given SplitIntoParagraphs function", t, func() {
		Convey("Should split text into paragraphs", func() {
			input := "Para 1\n\nPara 2\n\n\nPara 3"
			result := SplitIntoParagraphs(input)
			So(len(result), ShouldEqual, 3)
			So(result[0], ShouldEqual, "Para 1")
			So(result[1], ShouldEqual, "Para 2")
			So(result[2], ShouldEqual, "Para 3")
		})

		Convey("Should handle empty lines", func() {
			input := "\n\nPara 1\n\n\n\nPara 2\n\n"
			result := SplitIntoParagraphs(input)
			So(len(result), ShouldEqual, 2)
		})
	})
}

func TestSplitIntoSentences(t *testing.T) {
	Convey("Given SplitIntoSentences function", t, func() {
		Convey("Should split text into sentences", func() {
			input := "Hello there! How are you? I am fine."
			result := SplitIntoSentences(input)
			So(len(result), ShouldEqual, 3)
			So(result[0], ShouldEqual, "Hello there!")
			So(result[1], ShouldEqual, "How are you?")
			So(result[2], ShouldEqual, "I am fine.")
		})

		Convey("Should handle single sentence", func() {
			input := "This is one sentence."
			result := SplitIntoSentences(input)
			So(len(result), ShouldEqual, 1)
			So(result[0], ShouldEqual, "This is one sentence.")
		})

		Convey("Should handle text with abbreviations", func() {
			input := "Mr. Smith went to Washington D.C. by car. He had a meeting."
			result := SplitIntoSentences(input)
			So(len(result), ShouldEqual, 2)
			So(result[0], ShouldEqual, "Mr. Smith went to Washington D.C. by car.")
			So(result[1], ShouldEqual, "He had a meeting.")
		})

		Convey("Should handle multiple punctuation marks", func() {
			input := "Wow! That's amazing! Really?"
			result := SplitIntoSentences(input)
			So(len(result), ShouldEqual, 3)
			So(result[0], ShouldEqual, "Wow!")
			So(result[1], ShouldEqual, "That's amazing!")
			So(result[2], ShouldEqual, "Really?")
		})
	})
}

func TestWrapText(t *testing.T) {
	Convey("Given WrapText function", t, func() {
		Convey("Should wrap text to specified width", func() {
			input := "This is a long sentence that needs to be wrapped to multiple lines"
			result := WrapText(input, 20)
			So(len(result), ShouldBeGreaterThan, 1)
			for _, line := range result {
				So(len(line), ShouldBeLessThanOrEqualTo, 20)
			}
		})

		Convey("Should handle very long words", func() {
			input := "This supercalifragilisticexpialidocious word is very long"
			result := WrapText(input, 10)
			So(len(result), ShouldBeGreaterThan, 1)
		})
	})
}

func TestQuickWrapWithAttributes(t *testing.T) {
	Convey("Given QuickWrapWithAttributes function", t, func() {
		Convey("Should wrap content with attributes", func() {
			attrs := map[string]string{
				"id":    "123",
				"class": "test",
			}
			result := QuickWrapWithAttributes("div", "content", 0, attrs)
			So(result, ShouldContainSubstring, "<div")
			So(result, ShouldContainSubstring, "id=\"123\"")
			So(result, ShouldContainSubstring, "class=\"test\"")
		})
	})
}

func TestIndent(t *testing.T) {
	Convey("Given Indent function", t, func() {
		Convey("Should indent content with tabs", func() {
			result := Indent("content", 2)
			So(result, ShouldEqual, "\t\tcontent")
		})

		Convey("Should handle zero indent", func() {
			result := Indent("content", 0)
			So(result, ShouldEqual, "content")
		})
	})
}

func TestSubstitute(t *testing.T) {
	Convey("Given Substitute function", t, func() {
		Convey("And the indent is 0", func() {
			indent := 0

			Convey("Should replace placeholders with values", func() {
				fragments := map[string]string{
					"name": "Marvin",
				}
				result := Substitute("Hello {{name}}", fragments, indent)
				So(result, ShouldEqual, "Hello Marvin")
			})
		})

		Convey("And the indent is 2", func() {
			indent := 2

			Convey("Should replace placeholders with values", func() {
				fragments := map[string]string{
					"name": "Marvin",
				}
				result := Substitute("Hello {{name}}", fragments, indent)
				So(result, ShouldEqual, "\t\tHello Marvin")
			})
		})
	})
}
