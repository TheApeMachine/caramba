package neon

import (
	"fmt"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestOptimizerMixedPrecisionRegistration(t *testing.T) {
	optimizers := []struct {
		name       string
		bf16Inputs []dtype.DType
		bf16Output dtype.DType
	}{
		{"adam_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.Float32, dtype.Float32}, dtype.BFloat16},
		{"adamw_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.Float32, dtype.Float32}, dtype.BFloat16},
		{"adamax_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.Float32, dtype.Float32}, dtype.BFloat16},
		{"adagrad_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.Float32}, dtype.BFloat16},
		{"rmsprop_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.Float32}, dtype.BFloat16},
		{"lion_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.Float32}, dtype.BFloat16},
		{"sgd_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.Float32}, dtype.BFloat16},
		{"lars_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.Float32}, dtype.BFloat16},
		{"lbfgs_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16}, dtype.BFloat16},
		{"hebbian_step", []dtype.DType{dtype.BFloat16, dtype.BFloat16, dtype.BFloat16}, dtype.BFloat16},
	}

	for _, opt := range optimizers {
		t.Run(fmt.Sprintf("%s/bf16", opt.name), func(t *testing.T) {
			_, ok := Default.Lookup(opt.name, Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  opt.bf16Inputs,
				Outputs: []dtype.DType{opt.bf16Output},
			})

			if !ok {
				t.Fatalf("%s bf16 not registered", opt.name)
			}
		})

		fp16Inputs := make([]dtype.DType, len(opt.bf16Inputs))

		for index, dt := range opt.bf16Inputs {
			if dt == dtype.BFloat16 {
				fp16Inputs[index] = dtype.Float16
				continue
			}

			fp16Inputs[index] = dt
		}

		t.Run(fmt.Sprintf("%s/fp16", opt.name), func(t *testing.T) {
			_, ok := Default.Lookup(opt.name, Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  fp16Inputs,
				Outputs: []dtype.DType{dtype.Float16},
			})

			if !ok {
				t.Fatalf("%s fp16 not registered", opt.name)
			}
		})
	}
}
