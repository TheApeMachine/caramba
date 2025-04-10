package datura

import (
	"errors"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func (artifact *Artifact) DecryptPayload() (payload []byte, err error) {
	errnie.Debug("datura.Artifact.DecryptPayload")

	encryptedKey, err := artifact.EncryptedKey()
	if err != nil || len(encryptedKey) == 0 {
		return nil, errnie.Error(errors.New("missing or invalid encrypted key"), "key_err", err)
	}

	ephemeralPubKey, err := artifact.EphemeralPublicKey()
	if err != nil || len(ephemeralPubKey) == 0 {
		return nil, errnie.Error(errors.New("missing or invalid ephemeral public key"), "pubkey_err", err)
	}

	encryptedPayload, err := artifact.EncryptedPayload()
	if err != nil || len(encryptedPayload) == 0 {
		return nil, errnie.Error(errors.New("missing or invalid encrypted payload"), "payload_err", err)
	}

	crypto := NewCryptoSuite()
	payload, err = crypto.DecryptPayload(encryptedPayload, encryptedKey, ephemeralPubKey)

	if err != nil {
		return nil, errnie.Error(err, "payload", payload, "encryptedPayload", encryptedPayload, "encryptedKey", encryptedKey, "ephemeralPubKey", ephemeralPubKey)
	}

	return
}
