package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Dropout kernel — randomly zeros entries with probability config.Rate
and scales surviving entries by 1/(1-Rate) so the expected value is
preserved (inverted dropout, the standard training convention).
*/

type DropoutConfig struct {
	Rate float32
	Seed uint64
}

func DefaultDropoutConfig() DropoutConfig {
	return DropoutConfig{Rate: 0.1, Seed: 0xc0ffee}
}

func init() {
	Default.Register(Kernel{
		Name: "dropout",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runDropoutDefault,
	})
}

func runDropoutDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return DropoutFloat32(DefaultDropoutConfig(), args[0], args[1])
}

/*
DropoutFloat32 applies inverted dropout with the supplied config.
*/
func DropoutFloat32(config DropoutConfig, input, output tensor.Tensor) error {
	inView, err := input.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	if len(outView) != len(inView) {
		return tensor.ErrShapeMismatch
	}

	if config.Rate <= 0 {
		copy(outView, inView)
		return nil
	}

	keepProb := float32(1.0 - config.Rate)
	seedState := dropoutSeedState(config.Seed)
	dropoutFloat32Native(outView, inView, &seedState, keepProb)

	return nil
}

func dropoutSeedState(seed uint64) [4]uint32 {
	return [4]uint32{
		uint32(seed),
		uint32(seed >> 32),
		uint32(seed ^ 0x9e3779b9),
		uint32((seed >> 32) ^ 0x6c078965),
	}
}
