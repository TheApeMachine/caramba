package datura

import (
	"time"

	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
)

type ArtifactBuilder struct {
	artifact *Artifact
	crypto   *CryptoSuite
	err      error
}

func NewArtifactBuilder(
	mediatype MediaType,
	role ArtifactRole,
	scope ArtifactScope,
) *ArtifactBuilder {
	var (
		builder  = &ArtifactBuilder{}
		arena    = capnp.SingleSegment(nil)
		artifact Artifact
	)

	_, seg, err := capnp.NewMessage(arena)
	if err != nil {
		builder.err = err
		return builder
	}

	artifact, builder.err = NewRootArtifact(seg)
	if builder.err != nil {
		return nil
	}

	uuid, err := uuid.New().MarshalBinary()
	if err != nil {
		builder.err = err
		return builder
	}

	artifact.SetUuid(uuid)
	artifact.SetRole(uint32(role))
	artifact.SetScope(uint32(scope))
	artifact.SetMediatype(string(mediatype))
	artifact.SetTimestamp(time.Now().UnixNano())

	builder.artifact = &artifact

	// Initialize crypto suite
	builder.crypto = NewCryptoSuite()

	return builder
}

// SetPayload encrypts and sets the payload for the artifact
func (b *ArtifactBuilder) SetPayload(payload []byte) error {
	if b.err != nil {
		return b.err
	}

	// Encrypt the payload
	encryptedPayload, encryptedKey, ephemeralPubKey, err := b.crypto.EncryptPayload(payload)
	if err != nil {
		b.err = err
		return err
	}

	// Set the encrypted data in the artifact
	if err := (*b.artifact).SetEncryptedPayload(encryptedPayload); err != nil {
		b.err = err
		return err
	}

	if err := (*b.artifact).SetEncryptedKey(encryptedKey); err != nil {
		b.err = err
		return err
	}

	if err := (*b.artifact).SetEphemeralPublicKey(ephemeralPubKey); err != nil {
		b.err = err
		return err
	}

	return nil
}

// AddMetadata adds a key-value pair to the artifact's metadata
func (b *ArtifactBuilder) AddMetadata(key, value string) error {
	if b.err != nil {
		return b.err
	}

	metadata, err := (*b.artifact).NewMetadata(1)
	if err != nil {
		b.err = err
		return err
	}

	item := metadata.At(0)
	if err := item.SetKey(key); err != nil {
		b.err = err
		return err
	}

	if err := item.SetValue(value); err != nil {
		b.err = err
		return err
	}

	return nil
}

// Sign generates and sets a signature for the artifact
func (b *ArtifactBuilder) Sign(privateKey interface{}) error {
	if b.err != nil {
		return b.err
	}

	// For now, we'll just create a dummy signature
	// In a real implementation, this would use the private key
	signature := make([]byte, 64)
	if err := (*b.artifact).SetSignature(signature); err != nil {
		b.err = err
		return err
	}

	return nil
}

// Build finalizes and returns the artifact
func (b *ArtifactBuilder) Build() (*Artifact, error) {
	if b.err != nil {
		return nil, b.err
	}
	return b.artifact, nil
}

// Error returns any error that occurred during building
func (b *ArtifactBuilder) Error() error {
	return b.err
}
