package api

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestDeriveProjectSlug(t *testing.T) {
	Convey("Given deriveProjectSlug", t, func() {
		Convey("It should lowercase and hyphenate words", func() {
			So(deriveProjectSlug("My Cool Project"), ShouldEqual, "my-cool-project")
		})

		Convey("It should collapse punctuation into a single separator", func() {
			So(deriveProjectSlug("alpha---beta!!"), ShouldEqual, "alpha-beta")
		})

		Convey("It should fall back when the input has no slug characters", func() {
			So(deriveProjectSlug("   "), ShouldEqual, "project")
		})

		Convey("It should trim to the maximum slug length", func() {
			longName := "abcdefghijklmnopqrstuvwxyz-0123456789-abcdefghijklmnopqrstuvwxyz-0123456789-extra-tail"
			slug := deriveProjectSlug(longName)

			So(len(slug), ShouldBeLessThanOrEqualTo, 64)
			So(slug, ShouldNotContainSubstring, "-extra-tail")
		})
	})
}

func TestDedupeMemberIDs(t *testing.T) {
	Convey("Given dedupeMemberIDs", t, func() {
		Convey("It should always include the owner first without duplicates", func() {
			members := dedupeMemberIDs("owner-1", []string{"member-2", "owner-1", "member-3"})

			So(members, ShouldResemble, []string{"owner-1", "member-2", "member-3"})
		})
	})
}

func TestNormalizeProvisionPapers(t *testing.T) {
	Convey("Given normalizeProvisionPapers", t, func() {
		Convey("It should prefer explicit paper payloads over legacy paper_id", func() {
			papers := normalizeProvisionPapers(researchProjectProvisionRequest{
				PaperID: "legacy-id",
				Papers: []provisionPaperRequest{
					{ID: "paper-a", Title: "Main"},
					{ID: "paper-b", Title: "Appendix"},
				},
			})

			So(len(papers), ShouldEqual, 2)
			So(papers[0].ID, ShouldEqual, "paper-a")
			So(papers[1].Title, ShouldEqual, "Appendix")
		})

		Convey("It should map legacy paper_id when papers are omitted", func() {
			papers := normalizeProvisionPapers(researchProjectProvisionRequest{
				PaperID: "legacy-id",
				Name:    "Sparse attention",
			})

			So(papers, ShouldResemble, []provisionPaperRequest{{
				ID:    "legacy-id",
				Title: "Sparse attention",
			}})
		})
	})
}
