//go:build !darwin || !cgo

package metal

import "errors"

const metalUnavailableMsg = "metal backend unavailable: requires darwin and cgo"

var errMetalUnavailable = errors.New(metalUnavailableMsg)

func metalUnavailable() error {
	return errMetalUnavailable
}
