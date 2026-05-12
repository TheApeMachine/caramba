package devteam

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/config"
)

func TestFeatureBranch(t *testing.T) {
	Convey("Given a ColumnEvent with a feature title", t, func() {
		event := ColumnEvent{
			ID:    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
			Title: "Add dark mode support",
		}

		Convey("It should produce a slug-safe branch name", func() {
			branch := featureBranch(event)

			So(branch, ShouldStartWith, "devteam/")
			So(branch, ShouldContainSubstring, "f47ac10b")
			So(branch, ShouldContainSubstring, "add")
			So(branch, ShouldContainSubstring, "dark")
			So(branch, ShouldNotContainSubstring, " ")
		})
	})

	Convey("Given a very long feature title", t, func() {
		event := ColumnEvent{
			ID:    "aaaaaaaa-0000-0000-0000-000000000000",
			Title: "implement a very long feature that has many words in the title beyond limit",
		}

		Convey("It should truncate to at most 6 slug words", func() {
			branch := featureBranch(event)
			parts := splitBranchSlug(branch)
			// At most 6 words after the ID segment.
			So(len(parts), ShouldBeLessThanOrEqualTo, 6)
		})
	})
}

func TestIsRelevant(t *testing.T) {
	Convey("Given an Orchestrator configured for the requests project", t, func() {
		cfg := &config.DevTeamConfig{
			RequestsProjectID: "f47ac10b-58cc-4372-a567-0e02b2c3d479",
		}

		orchestrator := &Orchestrator{
			ctx: context.Background(),
			cfg: cfg,
			sem: make(chan struct{}, 1),
		}

		Convey("It should accept todo events on the requests project", func() {
			event := ColumnEvent{
				ColumnKey:         "todo",
				ResearchProjectID: "f47ac10b-58cc-4372-a567-0e02b2c3d479",
			}

			So(orchestrator.isRelevant(event), ShouldBeTrue)
		})

		Convey("It should reject events on a different column", func() {
			event := ColumnEvent{
				ColumnKey:         "in-progress",
				ResearchProjectID: "f47ac10b-58cc-4372-a567-0e02b2c3d479",
			}

			So(orchestrator.isRelevant(event), ShouldBeFalse)
		})

		Convey("It should reject events on a different project", func() {
			event := ColumnEvent{
				ColumnKey:         "todo",
				ResearchProjectID: "00000000-0000-0000-0000-000000000001",
			}

			So(orchestrator.isRelevant(event), ShouldBeFalse)
		})
	})
}

func BenchmarkFeatureBranch(b *testing.B) {
	event := ColumnEvent{
		ID:    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
		Title: "Add real-time collaboration to the kanban board",
	}

	for b.Loop() {
		_ = featureBranch(event)
	}
}

// splitBranchSlug splits the word segments after the ID prefix.
func splitBranchSlug(branch string) []string {
	// branch format: devteam/<8-char-id>-word1-word2-...
	// skip "devteam/" prefix (8 chars) + "/" + 8-char id + "-"
	if len(branch) < len("devteam/")+9 {
		return nil
	}

	rest := branch[len("devteam/")+9:]
	words := make([]string, 0)
	current := ""

	for _, ch := range rest {
		if ch == '-' {
			if current != "" {
				words = append(words, current)
				current = ""
			}
		} else {
			current += string(ch)
		}
	}

	if current != "" {
		words = append(words, current)
	}

	return words
}
