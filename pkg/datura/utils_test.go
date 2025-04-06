package datura

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestDecryptPayload(t *testing.T) {
	Convey("Given an artifact with encrypted payload", t, func() {
		originalPayload := []byte("test payload")
		artifact := New(
			WithMediatype(MediaTypeTextPlain),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeContext),
			WithEncryptedPayload(originalPayload),
		)

		Convey("When decrypting the payload", func() {
			decryptedPayload, err := artifact.DecryptPayload()

			Convey("Then it should succeed", func() {
				So(err, ShouldBeNil)
				So(decryptedPayload, ShouldResemble, originalPayload)
			})
		})
	})

	Convey("Given an artifact without encrypted payload", t, func() {
		artifact := New(
			WithMediatype(MediaTypeTextPlain),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeContext),
		)

		Convey("When attempting to decrypt non-existent payload", func() {
			decryptedPayload, err := artifact.DecryptPayload()

			Convey("Then it should fail", func() {
				So(err, ShouldNotBeNil)
				So(decryptedPayload, ShouldBeNil)
			})
		})
	})
}
