package notary

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNotary(t *testing.T) {
	Convey("Given a Notary and cryptographic identities", t, func() {
		notary := NewNotary()

		userA, err := NewIdentity()
		So(err, ShouldBeNil)
		userB, err := NewIdentity()
		So(err, ShouldBeNil)

		// Mint some initial credits to the users and escrow
		notary.Ledger().Mint(userA.Address(), 1000)
		notary.Ledger().Mint("escrow", 0)

		Convey("When a user submits a manifest with a valid signature and funds", func() {
			manifestData := []byte("compute intensive manifest")
			signature := userA.Sign(manifestData)
			cost := int64(100)

			hash, err := notary.SubmitManifest(userA, manifestData, signature, cost)

			Convey("It should succeed, deduct funds, and record the artifact", func() {
				So(err, ShouldBeNil)
				So(hash, ShouldNotBeEmpty)

				// User A should have 100 less credits
				So(notary.Ledger().BalanceOf(userA.Address()), ShouldEqual, 900)
				// Escrow should hold the 100 credits
				So(notary.Ledger().BalanceOf("escrow"), ShouldEqual, 100)

				// Artifact should be verified
				So(notary.Ledger().VerifyArtifact(hash, userA.Address()), ShouldBeTrue)
			})
		})

		Convey("When a user submits with invalid signature", func() {
			manifestData := []byte("compute intensive manifest")
			// User B signs it, but we pretend User A sent it (signature mismatch)
			signature := userB.Sign(manifestData)
			cost := int64(100)

			_, err := notary.SubmitManifest(userA, manifestData, signature, cost)

			Convey("It should fail with invalid signature", func() {
				So(err, ShouldEqual, ErrInvalidSignature)
				So(notary.Ledger().BalanceOf(userA.Address()), ShouldEqual, 1000) // Funds intact
			})
		})

		Convey("When a user submits without enough funds", func() {
			manifestData := []byte("massive job")
			signature := userA.Sign(manifestData)
			cost := int64(5000)

			_, err := notary.SubmitManifest(userA, manifestData, signature, cost)

			Convey("It should fail with insufficient funds", func() {
				So(err, ShouldEqual, ErrInsufficientFunds)
			})
		})

		Convey("When a compute job is settled successfully", func() {
			// First, user A escrows 50 credits
			notary.Ledger().Mint("escrow", 50)

			resultData := []byte("tensor output")
			workerSignature := userB.Sign(resultData)
			payout := int64(50)

			err := notary.SettleCompute(userA.Address(), userB, resultData, workerSignature, payout)

			Convey("It should transfer funds from escrow to the worker and record provenance", func() {
				So(err, ShouldBeNil)
				So(notary.Ledger().BalanceOf("escrow"), ShouldEqual, 0)
				So(notary.Ledger().BalanceOf(userB.Address()), ShouldEqual, 50)

				// Result artifact should belong to the owner (User A)
				hashBytes := sha256.Sum256(resultData)
				expectedHash := hex.EncodeToString(hashBytes[:])

				So(notary.Ledger().VerifyArtifact(expectedHash, userA.Address()), ShouldBeTrue)
			})
		})
	})
}
