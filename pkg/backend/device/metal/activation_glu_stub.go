//go:build !darwin || !cgo

package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func runMetalGLU(gate tensor.Tensor, up tensor.Tensor, out tensor.Tensor) error {
	_ = gate
	_ = up
	_ = out

	return tensor.ErrNeedsPlatformSetup
}
