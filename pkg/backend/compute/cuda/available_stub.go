//go:build !linux || !cgo || !cuda

package cuda

import "fmt"

func Available() error {
	return fmt.Errorf("%s", cudaTensorUnavailableMsg)
}
