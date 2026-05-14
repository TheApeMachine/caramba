//go:build !darwin || !cgo

package metal

import "fmt"

const metalUnavailableMsg = "metal backend unavailable: requires darwin and cgo"

var errMetalUnavailable = fmt.Errorf("%s", metalUnavailableMsg)

func metalUnavailable() error {
	return errMetalUnavailable
}
