package datura

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewCryptoSuite(t *testing.T) {
	Convey("When creating a new CryptoSuite", t, func() {
		cs := NewCryptoSuite()

		Convey("Then it should be properly initialized", func() {
			So(cs, ShouldNotBeNil)
			So(cs.curve, ShouldNotBeNil)
		})
	})
}

func TestGenerateEphemeralKeyPair(t *testing.T) {
	Convey("Given a CryptoSuite", t, func() {
		cs := NewCryptoSuite()

		Convey("When generating an ephemeral key pair", func() {
			key, err := cs.GenerateEphemeralKeyPair()

			Convey("Then it should succeed", func() {
				So(err, ShouldBeNil)
				So(key, ShouldNotBeNil)

				pubKey := key.PublicKey()
				So(pubKey, ShouldNotBeNil)

				pubKeyBytes := pubKey.Bytes()
				So(pubKeyBytes, ShouldNotBeEmpty)
			})
		})
	})
}

func TestEncryptDecryptPayload(t *testing.T) {
	Convey("Given a CryptoSuite", t, func() {
		cs := NewCryptoSuite()
		testPayload := []byte("test payload")

		Convey("When encrypting a payload", func() {
			encryptedPayload, encryptedKey, ephemeralPubKey, err := cs.EncryptPayload(testPayload)

			Convey("Then encryption should succeed", func() {
				So(err, ShouldBeNil)
				So(encryptedPayload, ShouldNotBeEmpty)
				So(encryptedKey, ShouldNotBeEmpty)
				So(ephemeralPubKey, ShouldNotBeEmpty)

				Convey("And when decrypting the payload", func() {
					decryptedPayload, err := cs.DecryptPayload(encryptedPayload, encryptedKey, ephemeralPubKey)

					Convey("Then decryption should succeed", func() {
						So(err, ShouldBeNil)
						So(decryptedPayload, ShouldResemble, testPayload)
					})
				})
			})
		})

		Convey("When decrypting with invalid data", func() {
			invalidPayload := []byte("invalid ciphertext")
			invalidKey := make([]byte, 16)
			invalidPubKey := []byte("invalid pub key bytes")

			_, err := cs.DecryptPayload(invalidPayload, invalidKey, invalidPubKey)

			Convey("Then it should fail", func() {
				So(err, ShouldNotBeNil)
			})
		})
	})
}
