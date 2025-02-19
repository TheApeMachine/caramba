package utils

import (
	"github.com/theapemachine/caramba/datura"
)

func DecryptPayload(artifact *datura.Artifact) ([]byte, error) {
	encryptedPayload, err := artifact.EncryptedPayload()

	if err != nil {
		return nil, err
	}

	encryptedKey, err := artifact.EncryptedKey()

	if err != nil {
		return nil, err
	}

	ephemeralPublicKey, err := artifact.EphemeralPublicKey()

	if err != nil {
		return nil, err
	}

	decryptedPayload, err := datura.NewCryptoSuite().DecryptPayload(
		encryptedPayload,
		encryptedKey,
		ephemeralPublicKey,
	)

	if err != nil {
		return nil, err
	}

	return decryptedPayload, nil
}
