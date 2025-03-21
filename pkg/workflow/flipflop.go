package workflow

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func NewFlipFlop(from io.ReadWriter, to io.ReadWriter) (err error) {
	if _, err = io.Copy(to, from); err != nil {
		return errnie.Error(err)
	}

	if _, err = io.Copy(from, to); err != nil {
		return errnie.Error(err)
	}

	return nil
}
