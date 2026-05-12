package notary

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"fmt"
)

/*
Identity represents a cryptographic identity in the Caramba network.
It consists of an Ed25519 key pair. The public key acts as the node's
address or unique identifier on the ledger.
*/
type Identity struct {
	PrivateKey ed25519.PrivateKey
	PublicKey  ed25519.PublicKey
}

/*
NewIdentity generates a new Ed25519 key pair for a node.
*/
func NewIdentity() (*Identity, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("notary: failed to generate identity: %w", err)
	}

	return &Identity{
		PrivateKey: priv,
		PublicKey:  pub,
	}, nil
}

/*
Address returns the hex-encoded string of the public key, which serves
as the unique address on the ledger.
*/
func (id *Identity) Address() string {
	return hex.EncodeToString(id.PublicKey)
}

/*
Sign creates a cryptographic signature of the given payload.
*/
func (id *Identity) Sign(payload []byte) []byte {
	return ed25519.Sign(id.PrivateKey, payload)
}

/*
Verify checks if the provided signature for the payload is valid
for the given public key address.
*/
func Verify(address string, payload, signature []byte) bool {
	pubKeyBytes, err := hex.DecodeString(address)
	if err != nil || len(pubKeyBytes) != ed25519.PublicKeySize {
		return false
	}

	return ed25519.Verify(ed25519.PublicKey(pubKeyBytes), payload, signature)
}
