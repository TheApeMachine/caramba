//go:build !darwin || !cgo

package metal

import "fmt"

const metalUnavailableMsg = "metal backend unavailable: requires darwin and cgo"

func metalUnavailable() error {
	return fmt.Errorf("%s", metalUnavailableMsg)
}
