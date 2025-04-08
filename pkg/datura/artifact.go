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

type ArtifactOption func(*Artifact)

func New(options ...ArtifactOption) *Artifact {
	errnie.Trace("artifact.New")

	var (
		arena    = capnp.SingleSegment(nil)
		seg      *capnp.Segment
		artifact Artifact
		err      error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if artifact, err = NewRootArtifact(seg); errnie.Error(err) != nil {
		return nil
	}

	if errnie.Error(artifact.SetUuid(uuid.New().String())) != nil {
		return nil
	}

	artifact.SetState(uint64(errnie.StateReady))

	artifact.SetTimestamp(time.Now().UnixNano())

	for _, option := range options {
		option(&artifact)
	}

	return Register(&artifact)
}

func WithPayload(payload []byte) ArtifactOption {
	errnie.Trace("artifact.WithPayload")

	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetPayload(payload)) != nil {
			return
		}

		artifact.SetState(uint64(errnie.StateReady))
	}
}

func WithEncryptedPayload(payload []byte) ArtifactOption {
	errnie.Trace("artifact.WithEncryptedPayload")

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

		artifact.SetState(uint64(errnie.StateReady))
	}
}

func WithMetadata(metadata map[string]any) ArtifactOption {
	errnie.Trace("artifact.WithMetadata")

	return func(artifact *Artifact) {
		var (
			mdList    Artifact_Metadata_List
			newMdList Artifact_Metadata_List
			err       error
		)

		if mdList, err = artifact.Metadata(); errnie.Error(err) != nil {
			return
		}

		if newMdList, err = artifact.NewMetadata(
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

		artifact.SetState(uint64(errnie.StateReady))
	}
}

func WithArtifact(artifact *Artifact) ArtifactOption {
	errnie.Trace("artifact.WithArtifact")

	return func(a *Artifact) {
		a.ToState(errnie.StateReady)
		errnie.Try(io.Copy(a, artifact))
		a.ToState(errnie.StateReady)
	}
}

func WithSignature(signature []byte) ArtifactOption {
	errnie.Trace("artifact.WithSignature")

	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetSignature(signature)) != nil {
			return
		}

		artifact.ToState(errnie.StateReady)
	}
}

func WithRole(role ArtifactRole) ArtifactOption {
	errnie.Trace("artifact.WithRole")

	return func(artifact *Artifact) {
		artifact.SetRole(uint32(role))
	}
}

func WithScope(scope ArtifactScope) ArtifactOption {
	errnie.Trace("artifact.WithScope")

	return func(artifact *Artifact) {
		artifact.SetScope(uint32(scope))
	}
}

func WithMediatype(mediatype MediaType) ArtifactOption {
	errnie.Trace("artifact.WithMediatype")

	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetMediatype(string(mediatype))) != nil {
			return
		}

	}
}

func WithMeta(key string, value any) ArtifactOption {
	errnie.Trace("artifact.WithMeta")

	return func(artifact *Artifact) {
		artifact.SetMetaValue(key, value)
	}
}

func WithError(err error) ArtifactOption {
	errnie.Trace("artifact.WithError")

	return func(artifact *Artifact) {
		WithEncryptedPayload([]byte(err.Error()))(artifact)
	}
}

func WithBytes(b []byte) ArtifactOption {
	errnie.Trace("artifact.WithBytes")

	return func(artifact *Artifact) {
		artifact.ToState(errnie.StateReady)
		if _, err := io.Copy(
			artifact, bytes.NewBuffer(b),
		); errnie.Error(err) != nil {
			return
		}

		artifact.ToState(errnie.StateReady)
	}
}
