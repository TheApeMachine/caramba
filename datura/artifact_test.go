package datura

import (
	"testing"

	"capnproto.org/go/capnp/v3"
	. "github.com/smartystreets/goconvey/convey"
)

func TestArtifactCreation(t *testing.T) {
	Convey("Given a new ArtifactBuilder", t, func() {
		// Create a new artifact builder
		builder := NewArtifactBuilder(
			MediaTypeTextPlain,
			ArtifactRoleUser,
			ArtifactScopePrompt,
		)

		Convey("When setting a payload and metadata", func() {
			// Test payload encryption and setting
			payload := []byte("Hello, this is a test payload!")
			err := builder.SetPayload(payload)
			So(err, ShouldBeNil)

			// Add some metadata
			err = builder.AddMetadata("test_key", "test_value")
			So(err, ShouldBeNil)

			// Sign the artifact (currently with dummy signature)
			err = builder.Sign(nil)
			So(err, ShouldBeNil)

			Convey("Then the artifact should be built successfully", func() {
				// Build the final artifact
				artifact, err := builder.Build()
				So(err, ShouldBeNil)
				So(artifact, ShouldNotBeNil)

				Convey("And it should have all required fields", func() {
					So(artifact.HasUuid(), ShouldBeTrue)
					So(artifact.HasEncryptedPayload(), ShouldBeTrue)
					So(artifact.HasEncryptedKey(), ShouldBeTrue)
					So(artifact.HasEphemeralPublicKey(), ShouldBeTrue)
					So(artifact.HasSignature(), ShouldBeTrue)

					Convey("And the payload should be decryptable", func() {
						encryptedPayload, err := artifact.EncryptedPayload()
						So(err, ShouldBeNil)
						So(encryptedPayload, ShouldNotBeEmpty)

						encryptedKey, err := artifact.EncryptedKey()
						So(err, ShouldBeNil)
						So(encryptedKey, ShouldNotBeEmpty)

						ephemeralPubKey, err := artifact.EphemeralPublicKey()
						So(err, ShouldBeNil)
						So(ephemeralPubKey, ShouldNotBeEmpty)

						crypto := NewCryptoSuite()
						decryptedPayload, err := crypto.DecryptPayload(encryptedPayload, encryptedKey, ephemeralPubKey)
						So(err, ShouldBeNil)
						So(decryptedPayload, ShouldResemble, payload)
					})

					Convey("And the metadata should be retrievable", func() {
						metadata, err := artifact.Metadata()
						So(err, ShouldBeNil)
						So(metadata.Len(), ShouldEqual, 1)

						item := metadata.At(0)
						key, err := item.Key()
						So(err, ShouldBeNil)
						So(key, ShouldEqual, "test_key")

						value, err := item.Value()
						So(err, ShouldBeNil)
						So(value, ShouldEqual, "test_value")
					})
				})
			})
		})
	})
}

func TestArtifactEncryption(t *testing.T) {
	Convey("Given an ArtifactBuilder with encryption capabilities", t, func() {
		builder := NewArtifactBuilder(
			MediaTypeTextPlain,
			ArtifactRoleUser,
			ArtifactScopePrompt,
		)

		Convey("When encrypting a payload", func() {
			payload := []byte("Hello, this is a test payload!")
			err := builder.SetPayload(payload)
			So(err, ShouldBeNil)

			artifact, err := builder.Build()
			So(err, ShouldBeNil)

			Convey("Then the encrypted fields should be properly set", func() {
				encryptedPayload, err := artifact.EncryptedPayload()
				So(err, ShouldBeNil)
				So(encryptedPayload, ShouldNotBeEmpty)

				encryptedKey, err := artifact.EncryptedKey()
				So(err, ShouldBeNil)
				So(encryptedKey, ShouldNotBeEmpty)

				ephemeralPubKey, err := artifact.EphemeralPublicKey()
				So(err, ShouldBeNil)
				So(ephemeralPubKey, ShouldNotBeEmpty)

				Convey("And the payload should be decryptable", func() {
					crypto := NewCryptoSuite()
					decryptedPayload, err := crypto.DecryptPayload(encryptedPayload, encryptedKey, ephemeralPubKey)
					So(err, ShouldBeNil)
					So(decryptedPayload, ShouldResemble, payload)
				})
			})
		})
	})
}

func TestArtifactMetadata(t *testing.T) {
	Convey("Given an ArtifactBuilder", t, func() {
		builder := NewArtifactBuilder(
			MediaTypeTextPlain,
			ArtifactRoleUser,
			ArtifactScopePrompt,
		)

		Convey("When adding metadata", func() {
			err := builder.AddMetadata("test_key", "test_value")
			So(err, ShouldBeNil)

			artifact, err := builder.Build()
			So(err, ShouldBeNil)

			Convey("Then the metadata should be retrievable", func() {
				metadata, err := artifact.Metadata()
				So(err, ShouldBeNil)
				So(metadata.Len(), ShouldEqual, 1)

				item := metadata.At(0)
				key, err := item.Key()
				So(err, ShouldBeNil)
				So(key, ShouldEqual, "test_key")

				value, err := item.Value()
				So(err, ShouldBeNil)
				So(value, ShouldEqual, "test_value")
			})
		})
	})
}

func TestArtifactWithCircuit(t *testing.T) {
	Convey("Given an artifact with circuit integration", t, func() {
		_, seg, err := capnp.NewMessage(capnp.SingleSegment(nil))
		So(err, ShouldBeNil)

		artifact, err := NewArtifact(seg)
		So(err, ShouldBeNil)

		Convey("When setting up for zero-knowledge proof", func() {
			pseudonymHash := []byte{1, 2, 3, 4}
			merkleRoot := []byte{9, 8, 7, 6}

			err = artifact.SetPseudonymHash(pseudonymHash)
			So(err, ShouldBeNil)

			err = artifact.SetMerkleRoot(merkleRoot)
			So(err, ShouldBeNil)

			Convey("Then the circuit-related fields should be properly set", func() {
				retrievedHash, err := artifact.PseudonymHash()
				So(err, ShouldBeNil)
				So(retrievedHash, ShouldResemble, pseudonymHash)

				retrievedRoot, err := artifact.MerkleRoot()
				So(err, ShouldBeNil)
				So(retrievedRoot, ShouldResemble, merkleRoot)

				Convey("And we can generate and verify proofs", func() {
					// Note: Actual proof generation and verification is tested in circuit_test.go
					// Here we just verify the fields are properly set for the circuit to use
					So(artifact.HasPseudonymHash(), ShouldBeTrue)
					So(artifact.HasMerkleRoot(), ShouldBeTrue)
				})
			})
		})
	})
}

func TestArtifactBuilder(t *testing.T) {
	Convey("Given a new ArtifactBuilder", t, func() {
		builder := NewArtifactBuilder(
			MediaTypeTextPlain,
			ArtifactRoleUser,
			ArtifactScopePrompt,
		)

		Convey("When creating a basic artifact", func() {
			artifact, err := builder.Build()
			So(err, ShouldBeNil)
			So(artifact, ShouldNotBeNil)

			Convey("Then it should have the required basic fields", func() {
				uuid, err := artifact.Uuid()
				So(err, ShouldBeNil)
				So(uuid, ShouldNotBeEmpty)

				mediatype, err := artifact.Mediatype()
				So(err, ShouldBeNil)
				So(mediatype, ShouldEqual, string(MediaTypeTextPlain))

				role := artifact.Role()
				So(role, ShouldEqual, uint32(ArtifactRoleUser))

				scope := artifact.Scope()
				So(scope, ShouldEqual, uint32(ArtifactScopePrompt))

				timestamp := artifact.Timestamp()
				So(timestamp, ShouldBeGreaterThan, 0)
			})
		})
	})
}
