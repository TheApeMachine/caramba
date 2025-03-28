package datura

import (
	"errors"
	"fmt"
	"time"

	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ArtifactOption func(*Artifact)

func New(options ...ArtifactOption) *Artifact {
	errnie.Debug("datura.New")

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

	for _, option := range options {
		option(&artifact)
	}

	return &artifact
}

func WithPayload(payload []byte) ArtifactOption {
	errnie.Debug("datura.WithPayload")

	if len(payload) == 0 {
		errnie.Error(errors.New("payload is empty"))
		return nil
	}

	return func(artifact *Artifact) {
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

		errnie.Error(artifact.SetEncryptedPayload(encryptedPayload))
		errnie.Error(artifact.SetEncryptedKey(encryptedKey))
		errnie.Error(artifact.SetEphemeralPublicKey(ephemeralPubKey))
	}
}

func WithMetadata(metadata map[string]any) ArtifactOption {
	errnie.Debug("datura.WithMetadata")

	return func(artifact *Artifact) {
		var (
			mdList    Artifact_Metadata_List
			newMdList Artifact_Metadata_List
			err       error
		)

		if mdList, err = artifact.Metadata(); errnie.Error(err) != nil {
			return
		}

		if newMdList, err = (*artifact).NewMetadata(
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

func WithSignature(signature []byte) ArtifactOption {
	errnie.Debug("datura.WithSignature")

	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetSignature(signature)) != nil {
			return
		}
	}
}

func WithRole(role ArtifactRole) ArtifactOption {
	errnie.Debug("datura.WithRole")

	return func(artifact *Artifact) {
		artifact.SetRole(uint32(role))
	}
}

func WithScope(scope ArtifactScope) ArtifactOption {
	errnie.Debug("datura.WithScope")

	return func(artifact *Artifact) {
		artifact.SetScope(uint32(scope))
	}
}

func WithMediatype(mediatype MediaType) ArtifactOption {
	errnie.Debug("datura.WithMediatype")

	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetMediatype(string(mediatype))) != nil {
			return
		}
	}
}

func WithMeta(key string, value any) ArtifactOption {
	errnie.Debug("datura.WithMeta")

	return func(artifact *Artifact) {
		artifact.SetMetaValue(key, value)
	}
}

func WithError(err error) ArtifactOption {
	errnie.Debug("datura.WithError")

	return func(artifact *Artifact) {
		WithPayload([]byte(err.Error()))(artifact)
	}
}
