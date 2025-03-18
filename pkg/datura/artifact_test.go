package datura

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestArtifactCreation(t *testing.T) {
	Convey("Given the functional options for artifact creation", t, func() {
		// Test payload
		payload := []byte("Hello, this is a test payload!")

		// Test metadata
		metadata := map[string]any{
			"test_key": "test_value",
		}

		// Dummy signature for testing
		signature := []byte("dummy-signature")

		Convey("When creating a new artifact with options", func() {
			// Create a new artifact using the functional options
			artifact := New(
				WithMediatype(MediaTypeTextPlain),
				WithRole(ArtifactRoleUser),
				WithScope(ArtifactScopePrompt),
				WithPayload(payload),
				WithMetadata(metadata),
				WithSignature(signature),
			)

			So(artifact, ShouldNotBeNil)

			Convey("Then the artifact should have all required fields", func() {
				uuid, err := artifact.Uuid()
				So(err, ShouldBeNil)
				So(uuid, ShouldNotBeEmpty)

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
					metadataList, err := artifact.Metadata()
					So(err, ShouldBeNil)
					So(metadataList.Len(), ShouldEqual, 1)

					item := metadataList.At(0)
					key, err := item.Key()
					So(err, ShouldBeNil)
					So(key, ShouldEqual, "test_key")

					value := item.Value()
					So(value, ShouldEqual, "test_value")
				})
			})
		})
	})
}

func TestArtifactEncryption(t *testing.T) {
	Convey("Given the functional options for artifact creation", t, func() {
		// Test payload
		payload := []byte("Hello, this is a test payload!")

		Convey("When creating an artifact with encrypted payload", func() {
			artifact := New(
				WithMediatype(MediaTypeTextPlain),
				WithRole(ArtifactRoleUser),
				WithScope(ArtifactScopePrompt),
				WithPayload(payload),
			)

			So(artifact, ShouldNotBeNil)

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
	Convey("Given the functional options for artifact creation", t, func() {
		// Test metadata
		metadata := map[string]any{
			"test_key": "test_value",
		}

		Convey("When creating an artifact with metadata", func() {
			artifact := New(
				WithMediatype(MediaTypeTextPlain),
				WithRole(ArtifactRoleUser),
				WithScope(ArtifactScopePrompt),
				WithMetadata(metadata),
			)

			So(artifact, ShouldNotBeNil)

			Convey("Then the metadata should be retrievable", func() {
				metadataList, err := artifact.Metadata()
				So(err, ShouldBeNil)
				So(metadataList.Len(), ShouldEqual, 1)

				item := metadataList.At(0)
				key, err := item.Key()
				So(err, ShouldBeNil)
				So(key, ShouldEqual, "test_key")

				value := item.Value()
				So(value, ShouldEqual, "test_value")
			})
		})
	})
}

func TestArtifactBasicFields(t *testing.T) {
	Convey("Given the functional options for artifact creation", t, func() {
		Convey("When creating a basic artifact", func() {
			artifact := New(
				WithMediatype(MediaTypeTextPlain),
				WithRole(ArtifactRoleUser),
				WithScope(ArtifactScopePrompt),
			)

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

func TestArtifactWithCircuit(t *testing.T) {
	Convey("Given an artifact with circuit integration", t, func() {
		artifact := New(
			WithMediatype(MediaTypeTextPlain),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopePrompt),
		)
		So(artifact, ShouldNotBeNil)

		Convey("When setting up for zero-knowledge proof", func() {
			pseudonymHash := []byte{1, 2, 3, 4}
			merkleRoot := []byte{9, 8, 7, 6}

			err := artifact.SetPseudonymHash(pseudonymHash)
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
