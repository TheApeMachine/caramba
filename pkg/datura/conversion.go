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
	var payload []byte

	if payload, err = artifact.DecryptPayload(); err != nil {
		return errnie.Error(err)
	}

	if err = json.Unmarshal(payload, v); err != nil {
		return errnie.Error(err, "payload", payload)
	}

	return nil
}

/*
Error is a convenience function to set an error as the payload of the artifact.
*/
func (artifact *Artifact) Error(e error) (err error) {
	WithError(errnie.New(
		errnie.WithError(e),
	))(artifact)

	return e
}
