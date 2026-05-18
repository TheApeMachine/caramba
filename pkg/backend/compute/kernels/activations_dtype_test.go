package kernels

import (
	"fmt"
	"math"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Sweep parity test for all generic-dispatched unary activations across
bf16 and fp16. Verifies the auto-registration via registerUnary covers
every name and that the widen-compute-narrow path matches the scalar
reference exactly at the dtype's representable precision.
*/

func TestUnaryActivationsBFloat16Dispatch(t *testing.T) {
	cases := []struct {
		name string
		op   func(float32) float32
	}{
		{"sigmoid", func(v float32) float32 { return 1 / (1 + float32(math.Exp(float64(-v)))) }},
		{"silu", func(v float32) float32 { return v / (1 + float32(math.Exp(float64(-v)))) }},
		{"tanh", func(v float32) float32 { return float32(math.Tanh(float64(v))) }},
		{"exp", func(v float32) float32 { return float32(math.Exp(float64(v))) }},
		{"log", func(v float32) float32 { return float32(math.Log(float64(v))) }},
		{"gelu", func(v float32) float32 {
			const sqrtTwo = 1.41421356237309504880
			return 0.5 * v * float32(1+math.Erf(float64(v)/sqrtTwo))
		}},
		{"softplus", func(v float32) float32 {
			return float32(math.Log1p(math.Exp(float64(v))))
		}},
		{"elu", func(v float32) float32 {
			if v > 0 {
				return v
			}
			return float32(math.Exp(float64(v))) - 1
		}},
	}

	const n = 64

	for _, kase := range cases {
		t.Run(fmt.Sprintf("bf16/%s", kase.name), func(t *testing.T) {
			input := randomBF16Slice(n, int64(kase.name[0]))

			// For log: avoid non-positive values.
			if kase.name == "log" {
				for index := range input {
					input[index] = dtype.BF16(uint16(input[index]) & 0x7FFF)

					if uint16(input[index]) == 0 {
						input[index] = dtype.NewBfloat16FromFloat32(1)
					}
				}
			}

			inTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.BFloat16)
			outTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.BFloat16)
			inView, _ := inTensor.BFloat16Native()
			copy(inView, input)

			kernel, ok := Default.Lookup(kase.name, Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{dtype.BFloat16},
				Outputs: []dtype.DType{dtype.BFloat16},
			})

			if !ok {
				t.Fatalf("kernel %q for bf16 not registered", kase.name)
			}

			if err := kernel.Run(inTensor, outTensor); err != nil {
				t.Fatal(err)
			}

			outView, _ := outTensor.BFloat16Native()

			for index := range input {
				value := (&input[index]).Float32()
				expected := dtype.NewBfloat16FromFloat32(kase.op(value))

				if uint16(expected) != uint16(outView[index]) {
					t.Fatalf("%s lane %d input=%g expected=0x%04x got=0x%04x",
						kase.name, index, value,
						uint16(expected), uint16(outView[index]),
					)
				}
			}
		})
	}
}

func TestUnaryActivationsFloat16Dispatch(t *testing.T) {
	cases := []struct {
		name string
		op   func(float32) float32
	}{
		{"sigmoid", func(v float32) float32 { return 1 / (1 + float32(math.Exp(float64(-v)))) }},
		{"silu", func(v float32) float32 { return v / (1 + float32(math.Exp(float64(-v)))) }},
		{"tanh", func(v float32) float32 { return float32(math.Tanh(float64(v))) }},
		{"gelu", func(v float32) float32 {
			const sqrtTwo = 1.41421356237309504880
			return 0.5 * v * float32(1+math.Erf(float64(v)/sqrtTwo))
		}},
	}

	const n = 64

	for _, kase := range cases {
		t.Run(fmt.Sprintf("fp16/%s", kase.name), func(t *testing.T) {
			input := randomF16Slice(n, int64(kase.name[0]))

			inTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float16)
			outTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float16)
			inView, _ := inTensor.Float16Native()
			copy(inView, input)

			kernel, ok := Default.Lookup(kase.name, Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{dtype.Float16},
				Outputs: []dtype.DType{dtype.Float16},
			})

			if !ok {
				t.Fatalf("kernel %q for fp16 not registered", kase.name)
			}

			if err := kernel.Run(inTensor, outTensor); err != nil {
				t.Fatal(err)
			}

			outView, _ := outTensor.Float16Native()

			for index := range input {
				value := input[index].Float32()
				expected := dtype.Fromfloat32(kase.op(value))

				if uint16(expected) != uint16(outView[index]) {
					t.Fatalf("%s lane %d input=%g expected=0x%04x got=0x%04x",
						kase.name, index, value,
						uint16(expected), uint16(outView[index]),
					)
				}
			}
		})
	}
}
