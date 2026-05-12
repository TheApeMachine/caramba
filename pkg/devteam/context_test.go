package devteam

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// fixture writes a minimal Go source file into a temp dir and returns the dir path.
func writeGoFixture(dir, rel, content string) {
	full := filepath.Join(dir, rel)
	_ = os.MkdirAll(filepath.Dir(full), 0o755)
	_ = os.WriteFile(full, []byte(content), 0o644)
}

func TestContextExtractorNewClose(t *testing.T) {
	Convey("Given a NewContextExtractor call", t, func() {
		extractor, err := NewContextExtractor()

		Convey("It should initialise without error", func() {
			So(err, ShouldBeNil)
			So(extractor, ShouldNotBeNil)
		})

		Convey("It should close cleanly", func() {
			So(func() { extractor.Close() }, ShouldNotPanic)
		})
	})
}

func TestContextExtractorExtract(t *testing.T) {
	Convey("Given a small Go repository fixture", t, func() {
		dir := t.TempDir()

		writeGoFixture(dir, "pkg/greeter/greeter.go", `package greeter

func Greet(name string) string {
	return "Hello, " + name
}

func Farewell(name string) string {
	return "Goodbye, " + name
}
`)
		writeGoFixture(dir, "pkg/main/main.go", `package main

import "greeter"

func main() {
	greeter.Greet("world")
}
`)

		extractor, err := NewContextExtractor()
		So(err, ShouldBeNil)

		defer extractor.Close()

		Convey("It should find symbols matching a keyword", func() {
			radius, err := extractor.Extract(dir, []string{"greet"}, 1)

			So(err, ShouldBeNil)
			So(radius, ShouldNotBeNil)
			So(len(radius.RootSymbols), ShouldBeGreaterThan, 0)

			names := make([]string, 0, len(radius.RootSymbols))

			for _, sym := range radius.RootSymbols {
				names = append(names, strings.ToLower(sym.Name))
			}

			So(names, ShouldContain, "greet")
		})

		Convey("It should not match symbols from an unrelated file", func() {
			// Write a completely separate package so there is no path overlap.
			writeGoFixture(dir, "pkg/unrelated/unrelated.go", `package unrelated

func UnrelatedFunc() {}
`)
			radius, err := extractor.Extract(dir, []string{"greet"}, 0)

			So(err, ShouldBeNil)

			for _, sym := range radius.RootSymbols {
				So(strings.ToLower(sym.Name), ShouldNotEqual, "unrelatedfunc")
			}
		})

		Convey("It should include the file containing the matched symbol in root symbols", func() {
			radius, err := extractor.Extract(dir, []string{"farewell"}, 0)

			So(err, ShouldBeNil)
			So(len(radius.RootSymbols), ShouldBeGreaterThan, 0)
			So(radius.RootSymbols[0].File, ShouldContainSubstring, "greeter.go")
		})

		Convey("It should return an empty blast radius for an unknown keyword", func() {
			radius, err := extractor.Extract(dir, []string{"xyznonexistent"}, 2)

			So(err, ShouldBeNil)
			So(len(radius.RootSymbols), ShouldEqual, 0)
			So(len(radius.ReachableFiles), ShouldEqual, 0)
		})
	})
}

func TestBlastRadiusFormat(t *testing.T) {
	Convey("Given a BlastRadius with symbols, files, and call graph entries", t, func() {
		radius := &BlastRadius{
			RootSymbols: []Symbol{
				{Name: "Greet", Kind: "function_declaration", File: "pkg/greeter/greeter.go", Line: 3},
			},
			ReachableFiles: []string{"pkg/main/main.go"},
			CallGraph: map[string][]Symbol{
				"Greet": {{Name: "main", File: "pkg/main/main.go", Line: 6}},
			},
		}

		Convey("It should produce a non-empty markdown block", func() {
			out := radius.Format()

			So(out, ShouldContainSubstring, "Blast Radius")
			So(out, ShouldContainSubstring, "`Greet`")
			So(out, ShouldContainSubstring, "pkg/main/main.go")
			So(out, ShouldContainSubstring, "`main`")
		})

		Convey("It should omit the files section when ReachableFiles is empty", func() {
			radius.ReachableFiles = nil
			out := radius.Format()

			So(out, ShouldNotContainSubstring, "Files in blast radius")
		})
	})
}

func TestMatchSymbols(t *testing.T) {
	Convey("Given a set of symbols and keywords", t, func() {
		symbols := []Symbol{
			{Name: "Greet", File: "greeter.go"},
			{Name: "Farewell", File: "greeter.go"},
			{Name: "RunServer", File: "server.go"},
		}

		Convey("It should match symbols whose name contains the keyword", func() {
			// "greet" matches "Greet" by name; "Farewell" and "RunServer" don't match.
			matched := matchSymbols(symbols, []string{"greet"})

			names := make([]string, len(matched))
			for i, sym := range matched {
				names[i] = sym.Name
			}

			So(names, ShouldContain, "Greet")
			So(names, ShouldNotContain, "RunServer")
		})

		Convey("It should match symbols whose file path contains the keyword", func() {
			matched := matchSymbols(symbols, []string{"server"})

			So(len(matched), ShouldEqual, 1)
			So(matched[0].Name, ShouldEqual, "RunServer")
		})

		Convey("It should return nothing for an unmatched keyword", func() {
			matched := matchSymbols(symbols, []string{"xyzzy"})

			So(len(matched), ShouldEqual, 0)
		})

		Convey("It should be case-insensitive", func() {
			matched := matchSymbols(symbols, []string{"FAREWELL"})

			So(len(matched), ShouldEqual, 1)
			So(matched[0].Name, ShouldEqual, "Farewell")
		})
	})
}

func BenchmarkContextExtractorExtract(b *testing.B) {
	dir := b.TempDir()

	for i := range 20 {
		writeGoFixture(dir, fmt.Sprintf("pkg/pkg%d/file.go", i), fmt.Sprintf(`package pkg%d

func FuncA%d() {}
func FuncB%d() {}
`, i, i, i))
	}

	extractor, _ := NewContextExtractor()
	defer extractor.Close()

	for b.Loop() {
		_, _ = extractor.Extract(dir, []string{"funca"}, 2)
	}
}
