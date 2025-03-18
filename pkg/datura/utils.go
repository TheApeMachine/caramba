package datura

func (artifact *Artifact) DecryptPayload() (payload []byte, err error) {
	encryptedKey, err := artifact.EncryptedKey()

	if err != nil {
		return nil, err
	}

	ephemeralPubKey, err := artifact.EphemeralPublicKey()

	if err != nil {
		return nil, err
	}

	encryptedPayload, err := artifact.EncryptedPayload()

	if err != nil {
		return nil, err
	}

	crypto := NewCryptoSuite()
	payload, err = crypto.DecryptPayload(encryptedPayload, encryptedKey, ephemeralPubKey)

	if err != nil {
		return nil, err
	}

	return
}
