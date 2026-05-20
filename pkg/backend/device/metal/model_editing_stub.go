//go:build !darwin || !cgo

package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func runMetalWeightGraftAddFloat32(weights tensor.Tensor, injection tensor.Tensor) error {
	_ = weights
	_ = injection

	return tensor.ErrNeedsPlatformSetup
}
