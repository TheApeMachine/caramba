package datura

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"time"

	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ArtifactOption func(Artifact) Artifact

func New(options ...ArtifactOption) Artifact {
	errnie.Trace("artifact.New")

	var (
		arena    = capnp.SingleSegment(nil)
		seg      *capnp.Segment
		artifact Artifact
		err      error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return errnie.Try(NewArtifact(seg)).ToState(errnie.StateError)
	}

	if artifact, err = NewRootArtifact(seg); errnie.Error(err) != nil {
		return errnie.Try(NewArtifact(seg)).ToState(errnie.StateError)
	}

	if errnie.Error(artifact.SetUuid(uuid.New().String())) != nil {
		return errnie.Try(NewArtifact(seg)).ToState(errnie.StateError)
	}

	artifact.SetTimestamp(time.Now().UnixNano())

	for _, option := range options {
		artifact = option(artifact)
	}

	return Register(artifact)
}

func WithPayload(payload []byte) ArtifactOption {
	errnie.Trace("artifact.WithPayload")

	return func(artifact Artifact) Artifact {
		if errnie.Error(artifact.SetPayload(payload)) != nil {
			return artifact
		}

		return artifact
	}
}

func WithEncryptedPayload(payload []byte) ArtifactOption {
	errnie.Trace("artifact.WithEncryptedPayload")

	if len(payload) == 0 {
		errnie.Error(errors.New("payload is empty"))
		return nil
	}

	return func(artifact Artifact) Artifact {
		var (
			crypto           = NewCryptoSuite()
			encryptedPayload []byte
			encryptedKey     []byte
			ephemeralPubKey  []byte
			err              error
		)

		if encryptedPayload, encryptedKey, ephemeralPubKey, err = crypto.EncryptPayload(payload); err != nil {
			errnie.Error(err)
			return artifact
		}

		errnie.Error(artifact.SetEncryptedPayload(encryptedPayload))
		errnie.Error(artifact.SetEncryptedKey(encryptedKey))
		errnie.Error(artifact.SetEphemeralPublicKey(ephemeralPubKey))

		return artifact
	}
}

func WithMetadata(metadata map[string]any) ArtifactOption {
	errnie.Trace("artifact.WithMetadata")

	return func(artifact Artifact) Artifact {
		var (
			mdList    Artifact_Metadata_List
			newMdList Artifact_Metadata_List
			err       error
		)

		if mdList, err = artifact.Metadata(); errnie.Error(err) != nil {
			return artifact
		}

		if newMdList, err = artifact.NewMetadata(
			int32(mdList.Len() + len(metadata)),
		); errnie.Error(err) != nil {
			return artifact
		}

		for idx := range mdList.Len() {
			if errnie.Error(newMdList.Set(idx, mdList.At(idx))) != nil {
				return artifact
			}
		}

		for key, value := range metadata {
			item := newMdList.At(newMdList.Len() - 1)

			if errnie.Error(item.SetKey(key)) != nil {
				return artifact
			}

			switch v := value.(type) {
			case string:
				if errnie.Error(item.Value().SetTextValue(v)) != nil {
					return artifact
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

		return artifact
	}
}

func WithArtifact(artifact Artifact) ArtifactOption {
	errnie.Trace("artifact.WithArtifact")

	return func(artifact Artifact) Artifact {
		artifact = errnie.Try(
			ReadRootArtifact(artifact.Message()),
		)

		return artifact
	}
}

func WithSignature(signature []byte) ArtifactOption {
	errnie.Trace("artifact.WithSignature")

	return func(artifact Artifact) Artifact {
		if errnie.Error(artifact.SetSignature(signature)) != nil {
			return artifact
		}

		return artifact
	}
}

func WithRole(role ArtifactRole) ArtifactOption {
	errnie.Trace("artifact.WithRole")

	return func(artifact Artifact) Artifact {
		artifact.SetRole(uint32(role))
		return artifact
	}
}

func WithScope(scope ArtifactScope) ArtifactOption {
	errnie.Trace("artifact.WithScope")

	return func(artifact Artifact) Artifact {
		artifact.SetScope(uint32(scope))
		return artifact
	}
}

func WithMediatype(mediatype MediaType) ArtifactOption {
	errnie.Trace("artifact.WithMediatype")

	return func(artifact Artifact) Artifact {
		if errnie.Error(artifact.SetMediatype(string(mediatype))) != nil {
			return artifact
		}

		return artifact
	}
}

func WithMeta(key string, value any) ArtifactOption {
	errnie.Trace("artifact.WithMeta")

	return func(artifact Artifact) Artifact {
		artifact.SetMetaValue(key, value)
		return artifact
	}
}

func WithError(err error) ArtifactOption {
	errnie.Trace("artifact.WithError")

	return func(artifact Artifact) Artifact {
		WithEncryptedPayload([]byte(err.Error()))(artifact)
		return artifact
	}
}

func WithBytes(b []byte) ArtifactOption {
	errnie.Trace("artifact.WithBytes")

	return func(artifact Artifact) Artifact {
		if _, err := io.Copy(
			artifact, bytes.NewBuffer(b),
		); errnie.Error(err) != nil {
			return artifact
		}

		return artifact
	}
}
