package datura

import (
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
	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetPayload(payload)) != nil {
			return
		}

		artifact.SetState(uint64(errnie.StateReady))
	}
}

func WithEncryptedPayload(payload []byte) ArtifactOption {
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
	return func(a *Artifact) {
		a.ToState(errnie.StateReady)
		errnie.Try(io.Copy(a, artifact))
		a.ToState(errnie.StateReady)
	}
}

func WithOrigin(origin string) ArtifactOption {
	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetOrigin(origin)) != nil {
			return
		}
	}
}

func WithIssuer(issuer string) ArtifactOption {
	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetIssuer(issuer)) != nil {
			return
		}
	}
}

func WithSignature(signature []byte) ArtifactOption {
	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetSignature(signature)) != nil {
			return
		}

		artifact.ToState(errnie.StateReady)
	}
}

func WithRole(role ArtifactRole) ArtifactOption {
	return func(artifact *Artifact) {
		artifact.SetRole(uint32(role))
	}
}

func WithScope(scope ArtifactScope) ArtifactOption {
	return func(artifact *Artifact) {
		artifact.SetScope(uint32(scope))
	}
}

func WithMediatype(mediatype MediaType) ArtifactOption {
	return func(artifact *Artifact) {
		if errnie.Error(artifact.SetMediatype(string(mediatype))) != nil {
			return
		}

	}
}

func WithMeta(key string, value any) ArtifactOption {
	return func(artifact *Artifact) {
		artifact.SetMetaValue(key, value)
	}
}

func WithError(err error) ArtifactOption {
	return func(artifact *Artifact) {
		WithEncryptedPayload([]byte(err.Error()))(artifact)
	}
}

func WithBytes(b []byte) ArtifactOption {
	return func(artifact *Artifact) {
		artifact.ToState(errnie.StateReady)

		msg, err := capnp.Unmarshal(b)
		if errnie.Error(err) != nil {
			return
		}

		decodedArtifact, err := ReadRootArtifact(msg)
		if errnie.Error(err) != nil {
			return
		}

		// Copy fields from decoded artifact to the target artifact
		if p, err := decodedArtifact.Payload(); err == nil {
			errnie.Error(artifact.SetPayload(p))
		} else {
			errnie.Error(err)
		}

		if uuidStr, err := decodedArtifact.Uuid(); err == nil {
			errnie.Error(artifact.SetUuid(uuidStr))
		} else {
			errnie.Error(err)
		}

		artifact.SetTimestamp(decodedArtifact.Timestamp())
		artifact.SetState(decodedArtifact.State())
		artifact.SetRole(decodedArtifact.Role())
		artifact.SetScope(decodedArtifact.Scope())

		if mt, err := decodedArtifact.Mediatype(); err == nil {
			errnie.Error(artifact.SetMediatype(mt))
		} else {
			errnie.Error(err)
		}

		if origin, err := decodedArtifact.Origin(); err == nil {
			errnie.Error(artifact.SetOrigin(origin))
		} else {
			errnie.Error(err)
		}

		if issuer, err := decodedArtifact.Issuer(); err == nil {
			errnie.Error(artifact.SetIssuer(issuer))
		} else {
			errnie.Error(err)
		}

		if sig, err := decodedArtifact.Signature(); err == nil {
			errnie.Error(artifact.SetSignature(sig))
		} else {
			errnie.Error(err)
		}

		if ep, err := decodedArtifact.EncryptedPayload(); err == nil {
			errnie.Error(artifact.SetEncryptedPayload(ep))
		} else {
			errnie.Error(err)
		}

		if ek, err := decodedArtifact.EncryptedKey(); err == nil {
			errnie.Error(artifact.SetEncryptedKey(ek))
		} else {
			errnie.Error(err)
		}

		if epk, err := decodedArtifact.EphemeralPublicKey(); err == nil {
			errnie.Error(artifact.SetEphemeralPublicKey(epk))
		} else {
			errnie.Error(err)
		}

		// Copy metadata
		if decodedMdList, err := decodedArtifact.Metadata(); err == nil {
			mdListLen := decodedMdList.Len()
			if mdListLen > 0 {
				newMdList, err := artifact.NewMetadata(int32(mdListLen))
				if errnie.Error(err) == nil {
					for i := 0; i < mdListLen; i++ {
						errnie.Error(newMdList.Set(i, decodedMdList.At(i)))
					}
				}
			}
		} else {
			errnie.Error(err)
		}

		artifact.ToState(errnie.StateReady)
	}
}
