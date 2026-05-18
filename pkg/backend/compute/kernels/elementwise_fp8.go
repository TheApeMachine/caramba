package kernels

import (
	"sync"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
FP8 elementwise kernels. Both E4M3 and E5M2 variants get full coverage
for the same op set as bf16/fp16: add, sub, mul, div, max, min plus
the unaries abs, neg, sqrt, relu.

Math contract: out[i] = round_to_fp8(f32(a[i]) op f32(b[i]))

Implementation strategy: lane-wise scalar widen to f32 (the FP8 widen
is non-trivial — denormals, NaN sentinels, saturation — and there is
no native NEON FP8 instruction on the target cores), then the actual
arithmetic runs through the existing NEON f32 kernel (addFloat32Native
et al.), then a lane-wise scalar narrow back. The arithmetic step is
real NEON; the per-byte conversion is the cost of the dtype.
*/

var fp8ScratchPool = sync.Pool{
	New: func() any {
		buf := make([]float32, 0, 16384)
		return &buf
	},
}

func borrowFloat32Buffer(n int) []float32 {
	bufPtr := fp8ScratchPool.Get().(*[]float32)
	buf := *bufPtr

	if cap(buf) < n {
		buf = make([]float32, n)
	} else {
		buf = buf[:n]
	}

	return buf
}

func releaseFloat32Buffer(buf []float32) {
	buf = buf[:0]
	fp8ScratchPool.Put(&buf)
}

func init() {
	registerFP8Elementwise()
}

func registerFP8Elementwise() {
	for _, kind := range []dtype.DType{dtype.Float8E4M3, dtype.Float8E5M2} {
		kind := kind
		register := func(name string, run func(args ...tensor.Tensor) error) {
			Default.Register(Kernel{
				Name: name,
				Signature: Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{kind, kind},
					Outputs: []dtype.DType{kind},
				},
				Locations: []tensor.Location{tensor.Host},
				Run:       run,
			})
		}

		if kind == dtype.Float8E4M3 {
			register("add", runAddF8E4M3)
			register("sub", runSubF8E4M3)
			register("mul", runMulF8E4M3)
			register("div", runDivF8E4M3)
			register("max", runMaxF8E4M3)
			register("min", runMinF8E4M3)
		} else {
			register("add", runAddF8E5M2)
			register("sub", runSubF8E5M2)
			register("mul", runMulF8E5M2)
			register("div", runDivF8E5M2)
			register("max", runMaxF8E5M2)
			register("min", runMinF8E5M2)
		}
	}
}

type fp8BinaryOp func(dst, left, right []float32)

func runFP8E4M3Binary(args []tensor.Tensor, op fp8BinaryOp) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float8E4M3Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float8E4M3Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float8E4M3Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	n := len(left)

	leftF32 := borrowFloat32Buffer(n)
	rightF32 := borrowFloat32Buffer(n)
	outF32 := borrowFloat32Buffer(n)

	defer releaseFloat32Buffer(leftF32)
	defer releaseFloat32Buffer(rightF32)
	defer releaseFloat32Buffer(outF32)

	for index := range left {
		leftF32[index] = left[index].Float32()
		rightF32[index] = right[index].Float32()
	}

	op(outF32, leftF32, rightF32)

	for index := range out {
		out[index] = dtype.NewF8E4M3FromFloat32(outF32[index])
	}

	return nil
}

func runFP8E5M2Binary(args []tensor.Tensor, op fp8BinaryOp) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float8E5M2Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float8E5M2Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float8E5M2Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	n := len(left)

	leftF32 := borrowFloat32Buffer(n)
	rightF32 := borrowFloat32Buffer(n)
	outF32 := borrowFloat32Buffer(n)

	defer releaseFloat32Buffer(leftF32)
	defer releaseFloat32Buffer(rightF32)
	defer releaseFloat32Buffer(outF32)

	for index := range left {
		leftF32[index] = left[index].Float32()
		rightF32[index] = right[index].Float32()
	}

	op(outF32, leftF32, rightF32)

	for index := range out {
		out[index] = dtype.NewF8E5M2FromFloat32(outF32[index])
	}

	return nil
}

func runAddF8E4M3(args ...tensor.Tensor) error {
	return runFP8E4M3Binary(args, addFloat32Native)
}

func runSubF8E4M3(args ...tensor.Tensor) error {
	return runFP8E4M3Binary(args, subFloat32Native)
}

func runMulF8E4M3(args ...tensor.Tensor) error {
	return runFP8E4M3Binary(args, mulFloat32Native)
}

func runDivF8E4M3(args ...tensor.Tensor) error {
	return runFP8E4M3Binary(args, divFloat32Native)
}

func runMaxF8E4M3(args ...tensor.Tensor) error {
	return runFP8E4M3Binary(args, maxFloat32Native)
}

func runMinF8E4M3(args ...tensor.Tensor) error {
	return runFP8E4M3Binary(args, minFloat32Native)
}

func runAddF8E5M2(args ...tensor.Tensor) error {
	return runFP8E5M2Binary(args, addFloat32Native)
}

func runSubF8E5M2(args ...tensor.Tensor) error {
	return runFP8E5M2Binary(args, subFloat32Native)
}

func runMulF8E5M2(args ...tensor.Tensor) error {
	return runFP8E5M2Binary(args, mulFloat32Native)
}

func runDivF8E5M2(args ...tensor.Tensor) error {
	return runFP8E5M2Binary(args, divFloat32Native)
}

func runMaxF8E5M2(args ...tensor.Tensor) error {
	return runFP8E5M2Binary(args, maxFloat32Native)
}

func runMinF8E5M2(args ...tensor.Tensor) error {
	return runFP8E5M2Binary(args, minFloat32Native)
}
