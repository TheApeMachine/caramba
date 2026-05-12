package devteam

import (
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSubtaskContextMarshal(t *testing.T) {
	Convey("Given a SubtaskContext with all fields populated", t, func() {
		snap := SubtaskContext{
			BlastRadius:  "### Blast Radius\n- `Greet` — greeter.go:3",
			KeySymbols:   []string{"Greet", "Farewell"},
			FilesInScope: []string{"pkg/greeter/greeter.go", "pkg/main/main.go"},
			SiblingNotes: map[string]string{
				"Add Farewell method": "also modifies pkg/greeter/greeter.go",
			},
		}

		Convey("It should round-trip through JSON without loss", func() {
			raw, err := json.Marshal(snap)

			So(err, ShouldBeNil)
			So(len(raw), ShouldBeGreaterThan, 0)

			var decoded SubtaskContext
			err = json.Unmarshal(raw, &decoded)

			So(err, ShouldBeNil)
			So(decoded.BlastRadius, ShouldEqual, snap.BlastRadius)
			So(decoded.KeySymbols, ShouldResemble, snap.KeySymbols)
			So(decoded.FilesInScope, ShouldResemble, snap.FilesInScope)
			So(decoded.SiblingNotes, ShouldResemble, snap.SiblingNotes)
		})
	})
}

func TestSubtaskContextEmpty(t *testing.T) {
	Convey("Given an empty SubtaskContext", t, func() {
		snap := SubtaskContext{}

		Convey("It should marshal to a valid JSON object", func() {
			raw, err := json.Marshal(snap)

			So(err, ShouldBeNil)

			var decoded SubtaskContext
			err = json.Unmarshal(raw, &decoded)

			So(err, ShouldBeNil)
			So(decoded.BlastRadius, ShouldBeEmpty)
			So(len(decoded.KeySymbols), ShouldEqual, 0)
		})
	})
}

func TestFormatSubtaskContext(t *testing.T) {
	Convey("Given a Subtask with a fully populated ContextSnapshot", t, func() {
		subtask := Subtask{
			Title: "Add Greet method",
			ContextSnapshot: SubtaskContext{
				BlastRadius:  "### Blast Radius\n- `Greet`",
				KeySymbols:   []string{"Greet"},
				FilesInScope: []string{"pkg/greeter/greeter.go"},
				SiblingNotes: map[string]string{
					"Add Farewell method": "also modifies greeter.go",
				},
			},
		}

		Convey("It should include all sections in the formatted output", func() {
			out := formatSubtaskContext(subtask)

			So(out, ShouldContainSubstring, "Blast Radius")
			So(out, ShouldContainSubstring, "pkg/greeter/greeter.go")
			So(out, ShouldContainSubstring, "`Greet`")
			So(out, ShouldContainSubstring, "Add Farewell method")
			So(out, ShouldContainSubstring, "also modifies greeter.go")
		})
	})

	Convey("Given a Subtask with an empty ContextSnapshot", t, func() {
		subtask := Subtask{Title: "Empty subtask"}

		Convey("It should return an empty string", func() {
			out := formatSubtaskContext(subtask)

			So(out, ShouldBeEmpty)
		})
	})
}

func TestKeywordsFromCard(t *testing.T) {
	Convey("Given a card title and description", t, func() {
		Convey("It should extract unique lowercase tokens of length >= 3", func() {
			keywords := keywordsFromCard(
				"Add Greet method",
				"Implement a Greet function in the greeter package",
			)

			So(keywords, ShouldContain, "add")
			So(keywords, ShouldContain, "greet")
			So(keywords, ShouldContain, "method")
			So(keywords, ShouldContain, "implement")
			So(keywords, ShouldContain, "greeter")
			So(keywords, ShouldContain, "package")
		})

		Convey("It should deduplicate repeated tokens", func() {
			keywords := keywordsFromCard("Greet greet GREET", "greet again")

			count := 0
			for _, kw := range keywords {
				if kw == "greet" {
					count++
				}
			}

			So(count, ShouldEqual, 1)
		})

		Convey("It should exclude tokens shorter than 3 characters", func() {
			keywords := keywordsFromCard("Add a Go test", "in it")

			for _, kw := range keywords {
				So(len(kw), ShouldBeGreaterThanOrEqualTo, 3)
			}
		})
	})
}

func BenchmarkSubtaskContextMarshal(b *testing.B) {
	snap := SubtaskContext{
		BlastRadius:  "### Blast Radius\n- `Greet` — greeter.go:3\n- `Farewell` — greeter.go:7",
		KeySymbols:   []string{"Greet", "Farewell", "RunServer"},
		FilesInScope: []string{"pkg/greeter/greeter.go", "pkg/main/main.go", "pkg/server/server.go"},
		SiblingNotes: map[string]string{
			"Add Farewell method": "also modifies pkg/greeter/greeter.go",
			"Wire HTTP handler":   "also modifies pkg/server/server.go",
		},
	}

	for b.Loop() {
		raw, _ := json.Marshal(snap)
		var decoded SubtaskContext
		_ = json.Unmarshal(raw, &decoded)
	}
}

func BenchmarkKeywordsFromCard(b *testing.B) {
	title := "Implement real-time collaborative editing for the kanban board"
	desc := "Add WebSocket support and operational transform logic to handle concurrent edits"

	for b.Loop() {
		_ = keywordsFromCard(title, desc)
	}
}
