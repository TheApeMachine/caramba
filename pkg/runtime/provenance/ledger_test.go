package provenance

import (
	"crypto/ed25519"
	"crypto/rand"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestLedgerRecordingAndDigest(t *testing.T) {
	Convey("Given a fresh ledger", t, func() {
		ledger := New(map[string]any{
			"program": "chat",
			"backend": "metal",
		})

		ledger.RecordProgram("chat", []byte("program bytes"))
		ledger.RecordAsset("model", "openai-community/gpt2", "abc123")
		ledger.RecordSeed("main", 42)
		ledger.RecordOutput("transcript", "/tmp/out.txt", "hashvalue")

		Convey("Entries should preserve insertion order", func() {
			entries := ledger.Entries()
			So(len(entries), ShouldEqual, 4)
			So(entries[0].Kind, ShouldEqual, "program")
			So(entries[1].Kind, ShouldEqual, "asset")
			So(entries[2].Kind, ShouldEqual, "seed")
			So(entries[3].Kind, ShouldEqual, "output")
		})

		Convey("Serialize should produce parseable JSON", func() {
			payload, err := ledger.Serialize()
			So(err, ShouldBeNil)
			So(len(payload), ShouldBeGreaterThan, 0)
		})

		Convey("Digest should be deterministic for the same ledger", func() {
			digest1, err := ledger.Digest()
			So(err, ShouldBeNil)

			digest2, err := ledger.Digest()
			So(err, ShouldBeNil)

			So(digest1, ShouldResemble, digest2)
			So(len(digest1), ShouldEqual, 32)
		})
	})
}

func TestLedgerSigningAndVerify(t *testing.T) {
	Convey("Given a ledger and an Ed25519 key pair", t, func() {
		publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
		So(err, ShouldBeNil)

		ledger := New(map[string]any{"program": "chat"})
		ledger.RecordProgram("chat", []byte("body"))

		Convey("Sign + Verify with matching key should pass", func() {
			signature, err := ledger.Sign(privateKey)
			So(err, ShouldBeNil)

			valid, err := ledger.Verify(publicKey, signature)
			So(err, ShouldBeNil)
			So(valid, ShouldBeTrue)
		})

		Convey("Verify with a different key should fail", func() {
			signature, err := ledger.Sign(privateKey)
			So(err, ShouldBeNil)

			otherPub, _, _ := ed25519.GenerateKey(rand.Reader)
			valid, err := ledger.Verify(otherPub, signature)
			So(err, ShouldBeNil)
			So(valid, ShouldBeFalse)
		})
	})
}

func TestLedgerWriteAndRead(t *testing.T) {
	Convey("A ledger written to disk should round-trip back", t, func() {
		path := filepath.Join(t.TempDir(), "ledger.json")

		original := New(map[string]any{"program": "diffusion"})
		original.RecordSeed("latents", 99)
		original.RecordOutput("image", "/tmp/img.png", "imghash")

		So(original.WriteFile(path), ShouldBeNil)

		loaded, err := ReadFile(path)
		So(err, ShouldBeNil)

		entries := loaded.Entries()
		So(len(entries), ShouldEqual, 2)
		So(entries[0].Seed, ShouldEqual, 99)
		So(entries[1].Path, ShouldEqual, "/tmp/img.png")

		Convey("Digest of the reloaded ledger should match the original", func() {
			originalDigest, _ := original.Digest()
			loadedDigest, _ := loaded.Digest()
			So(loadedDigest, ShouldResemble, originalDigest)
		})
	})
}
