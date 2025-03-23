package datura

import (
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
To is a convenience function to convert the artifact's payload into some
other type by unmarshalling it into the provided type.
*/
func (artifact *Artifact) To(v any) (err error) {
	errnie.Debug("datura.To")

	var payload []byte

	if payload, err = artifact.DecryptPayload(); err != nil {
		return errnie.Error(err, "payload", payload)
	}

	if err = json.Unmarshal(payload, v); err != nil {
		return errnie.Error(err, "payload", payload)
	}

	return nil
}

/*
From is a convenience function to set the artifact's payload from some
other type by marshalling it into the artifact's payload.
*/
func (artifact *Artifact) From(v any) (err error) {
	errnie.Debug("datura.From")

	var payload []byte

	if payload, err = json.Marshal(v); err != nil {
		return errnie.Error(err, "payload", string(payload))
	}

	WithPayload(payload)(artifact)
	return nil
}

/*
Error is a convenience function to set an error as the payload of the artifact.
*/
func (artifact *Artifact) Error(e error) (err error) {
	errnie.Debug("datura.Error", "e", e.Error())

	WithError(errnie.New(
		errnie.WithError(e),
	))(artifact)

	return e
}
