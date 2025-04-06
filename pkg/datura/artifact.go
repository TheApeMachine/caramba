package datura

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"time"

	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ArtifactState uint

const (
	ArtifactStateCreated ArtifactState = iota
	ArtifactStateBuffered
	ArtifactStateRead
)

type ArtifactBuilder struct {
	*Artifact
	encoder *capnp.Encoder
	decoder *capnp.Decoder
	buffer  *bufio.ReadWriter
	state   ArtifactState
}

type ArtifactBuilderOption func(*ArtifactBuilder)

func New(options ...ArtifactBuilderOption) *ArtifactBuilder {
	var (
		arena    = capnp.SingleSegment(nil)
		seg      *capnp.Segment
		artifact Artifact
		uid      []byte
		err      error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if artifact, err = NewRootArtifact(seg); errnie.Error(err) != nil {
		return nil
	}

	if uid, err = uuid.New().MarshalBinary(); errnie.Error(err) != nil {
		return nil
	}

	if errnie.Error(artifact.SetUuid(uid)) != nil {
		return nil
	}

	artifact.SetTimestamp(time.Now().UnixNano())

	shared := bytes.NewBuffer(nil)
	buffer := bufio.NewReadWriter(
		bufio.NewReader(shared),
		bufio.NewWriter(shared),
	)

	builder := &ArtifactBuilder{
		Artifact: &artifact,
		encoder:  capnp.NewEncoder(buffer),
		decoder:  capnp.NewDecoder(buffer),
		buffer:   buffer,
		state:    ArtifactStateCreated,
	}

	for _, option := range options {
		option(builder)
	}

	return builder
}

func WithPayload(payload []byte) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		if errnie.Error(builder.SetPayload(payload)) != nil {
			return
		}
	}
}

func WithEncryptedPayload(payload []byte) ArtifactBuilderOption {
	if len(payload) == 0 {
		errnie.Error(errors.New("payload is empty"))
		return nil
	}

	return func(builder *ArtifactBuilder) {
		var (
			crypto           = NewCryptoSuite()
			encryptedPayload []byte
			encryptedKey     []byte
			ephemeralPubKey  []byte
			err              error
		)

		if encryptedPayload, encryptedKey, ephemeralPubKey, err = crypto.EncryptPayload(payload); err != nil {
			errnie.Error(err)
			return
		}

		errnie.Error(builder.SetEncryptedPayload(encryptedPayload))
		errnie.Error(builder.SetEncryptedKey(encryptedKey))
		errnie.Error(builder.SetEphemeralPublicKey(ephemeralPubKey))
	}
}

func WithMetadata(metadata map[string]any) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		var (
			mdList    Artifact_Metadata_List
			newMdList Artifact_Metadata_List
			err       error
		)

		if mdList, err = builder.Metadata(); errnie.Error(err) != nil {
			return
		}

		if newMdList, err = builder.NewMetadata(
			int32(mdList.Len() + len(metadata)),
		); errnie.Error(err) != nil {
			return
		}

		for idx := range mdList.Len() {
			if errnie.Error(newMdList.Set(idx, mdList.At(idx))) != nil {
				return
			}
		}

		for key, value := range metadata {
			item := newMdList.At(newMdList.Len() - 1)

			if errnie.Error(item.SetKey(key)) != nil {
				return
			}

			switch v := value.(type) {
			case string:
				if errnie.Error(item.Value().SetTextValue(v)) != nil {
					return
				}
			case int:
				item.Value().SetIntValue(int64(v))
			case int64:
				item.Value().SetIntValue(v)
			case float64:
				item.Value().SetFloatValue(v)
			case bool:
				item.Value().SetBoolValue(v)
			case []byte:
				item.Value().SetBinaryValue(v)
			default:
				item.Value().SetTextValue(fmt.Sprintf("%v", v))
			}
		}
	}
}

func WithArtifact(artifact *Artifact) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		builder.Artifact = artifact
	}
}

func WithSignature(signature []byte) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		if errnie.Error(builder.SetSignature(signature)) != nil {
			return
		}
	}
}

func WithRole(role ArtifactRole) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		builder.SetRole(uint32(role))
	}
}

func WithScope(scope ArtifactScope) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		builder.SetScope(uint32(scope))
	}
}

func WithMediatype(mediatype MediaType) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		if errnie.Error(builder.SetMediatype(string(mediatype))) != nil {
			return
		}
	}
}

func WithMeta(key string, value any) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		builder.SetMetaValue(key, value)
	}
}

func WithError(err error) ArtifactBuilderOption {
	return func(builder *ArtifactBuilder) {
		WithEncryptedPayload([]byte(err.Error()))(builder)
	}
}
