package datura

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestArtifactIs(t *testing.T) {
	Convey("Given an artifact with specific role and scope", t, func() {
		artifact := New(
			WithMediatype(MediaTypeTextPlain),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeContext),
		)

		Convey("When checking matching role and scope", func() {
			result := artifact.Is(ArtifactRoleUser, ArtifactScopeContext)

			Convey("Then it should return true", func() {
				So(result, ShouldBeTrue)
			})
		})

		Convey("When checking non-matching role", func() {
			result := artifact.Is(ArtifactRoleSystem, ArtifactScopeContext)

			Convey("Then it should return false", func() {
				So(result, ShouldBeFalse)
			})
		})

		Convey("When checking non-matching scope", func() {
			result := artifact.Is(ArtifactRoleUser, ArtifactScopeGeneration)

			Convey("Then it should return false", func() {
				So(result, ShouldBeFalse)
			})
		})

		Convey("When checking non-matching role and scope", func() {
			result := artifact.Is(ArtifactRoleSystem, ArtifactScopeGeneration)

			Convey("Then it should return false", func() {
				So(result, ShouldBeFalse)
			})
		})
	})
}

func TestMediaTypeConstants(t *testing.T) {
	Convey("Given MediaType constants", t, func() {
		Convey("Then they should have correct string values", func() {
			So(string(MediaTypeUnknown), ShouldEqual, "unknown")
			So(string(MediaTypeTextPlain), ShouldEqual, "text/plain")
			So(string(MediaTypeApplicationJson), ShouldEqual, "application/json")
			So(string(MediaTypeApplicationYaml), ShouldEqual, "application/yaml")
			So(string(MediaTypeApplicationXml), ShouldEqual, "application/xml")
			So(string(MediaTypeApplicationPdf), ShouldEqual, "application/pdf")
			So(string(MediaTypeApplicationOctetStream), ShouldEqual, "application/octet-stream")
			So(string(MediaTypeCapnp), ShouldEqual, "application/capnp")
			So(string(MediaTypeApplicationZip), ShouldEqual, "application/zip")
			So(string(MediaTypeApplicationGzip), ShouldEqual, "application/gzip")
			So(string(MediaTypeApplicationXZip), ShouldEqual, "application/x-zip-compressed")
		})
	})
}
