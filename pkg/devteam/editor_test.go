package devteam

import (
	"fmt"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestApplyLineEdit(t *testing.T) {
	Convey("Given a multi-line file content", t, func() {
		content := strings.Join([]string{
			"package foo",
			"",
			"func Hello() string {",
			`	return "hello"`,
			"}",
		}, "\n")

		Convey("It should replace a matching block with new lines", func() {
			updated, err := applyLineEdit(
				content,
				[]string{`	return "hello"`},
				[]string{`	return "hello, world"`},
			)

			So(err, ShouldBeNil)
			So(updated, ShouldContainSubstring, `return "hello, world"`)
			So(updated, ShouldNotContainSubstring, `return "hello"`)
		})

		Convey("It should replace a multi-line block correctly", func() {
			updated, err := applyLineEdit(
				content,
				[]string{
					"func Hello() string {",
					`	return "hello"`,
					"}",
				},
				[]string{
					"func Hello() string {",
					`	return "greetings"`,
					"}",
				},
			)

			So(err, ShouldBeNil)
			So(updated, ShouldContainSubstring, "greetings")
			So(updated, ShouldNotContainSubstring, `"hello"`)
			So(updated, ShouldContainSubstring, "package foo")
		})

		Convey("It should return an error when the old block is not found", func() {
			_, err := applyLineEdit(
				content,
				[]string{"func NonExistent() {}"},
				[]string{"func Replacement() {}"},
			)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "not found")
		})

		Convey("It should return an error for an empty old_lines block", func() {
			_, err := applyLineEdit(content, nil, []string{"new line"})

			So(err, ShouldNotBeNil)
		})

		Convey("It should tolerate trailing whitespace differences in old_lines", func() {
			updated, err := applyLineEdit(
				content,
				[]string{"func Hello() string {   "}, // trailing spaces
				[]string{"func Hello() string { // edited"},
			)

			So(err, ShouldBeNil)
			So(updated, ShouldContainSubstring, "// edited")
		})
	})
}

func TestFindBlock(t *testing.T) {
	Convey("Given a slice of file lines", t, func() {
		lines := []string{"a", "b", "c", "d", "e"}

		Convey("It should return the correct index for a matching block at the start", func() {
			So(findBlock(lines, []string{"a", "b"}), ShouldEqual, 0)
		})

		Convey("It should return the correct index for a matching block in the middle", func() {
			So(findBlock(lines, []string{"c", "d"}), ShouldEqual, 2)
		})

		Convey("It should return the correct index for a single-line block at the end", func() {
			So(findBlock(lines, []string{"e"}), ShouldEqual, 4)
		})

		Convey("It should return -1 when the block is not present", func() {
			So(findBlock(lines, []string{"x", "y"}), ShouldEqual, -1)
		})

		Convey("It should return -1 for an empty block", func() {
			So(findBlock(lines, nil), ShouldEqual, -1)
		})

		Convey("It should return -1 when the block is longer than the file", func() {
			So(findBlock(lines, []string{"a", "b", "c", "d", "e", "f"}), ShouldEqual, -1)
		})
	})
}

func TestFirstOrEmpty(t *testing.T) {
	Convey("Given firstOrEmpty", t, func() {
		Convey("It should return the first element of a non-empty slice", func() {
			So(firstOrEmpty([]string{"hello", "world"}), ShouldEqual, "hello")
		})

		Convey("It should return an empty string for a nil slice", func() {
			So(firstOrEmpty(nil), ShouldBeEmpty)
		})

		Convey("It should return an empty string for an empty slice", func() {
			So(firstOrEmpty([]string{}), ShouldBeEmpty)
		})
	})
}

func BenchmarkApplyLineEdit(b *testing.B) {
	lines := make([]string, 200)
	for i := range lines {
		lines[i] = fmt.Sprintf("line %d content here", i)
	}

	content := strings.Join(lines, "\n")
	oldLines := []string{"line 100 content here"}
	newLines := []string{"line 100 replaced content"}

	for b.Loop() {
		_, _ = applyLineEdit(content, oldLines, newLines)
	}
}

func BenchmarkFindBlock(b *testing.B) {
	lines := make([]string, 500)
	for i := range lines {
		lines[i] = fmt.Sprintf("line-%d", i)
	}

	block := []string{"line-490", "line-491", "line-492"}

	for b.Loop() {
		_ = findBlock(lines, block)
	}
}
