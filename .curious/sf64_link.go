//go:build ignore

package main

import (
	"fmt"
	"math"
	_ "unsafe"

	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

//go:linkname fadd64 runtime.fadd64
func fadd64(x, y uint64) uint64

//go:linkname fmul64 runtime.fmul64
func fmul64(x, y uint64) uint64

//go:linkname fdiv64 runtime.fdiv64
func fdiv64(x, y uint64) uint64

//go:linkname f32to64 runtime.f32to64
func f32to64(x uint32) uint64

//go:linkname f64to32 runtime.f64to32
func f64to32(x uint64) uint32

func geluSF(value float32) float32 {
	value64 := f32to64(math.Float32bits(value))
	sqrtTwoInv := math.Float64bits(0.70710678118654752440)
	erfArg := fadd64(value64, fmul64(value64, sqrtTwoInv)) // WRONG - should be mul only
	_ = erfArg
	return 0
}

func main() {
	x := float32(1.0/12.0 - 4.0)
	ref := cpumath.FastGelu32(x)
	fmt.Println("x", x, "ref", ref)

	value64 := f32to64(math.Float32bits(x))
	sqrtTwoInv := math.Float64bits(0.70710678118654752440)
	erfArg := fmul64(value64, sqrtTwoInv)
	erfVal := math.Float64bits(math.Erf(math.Float64frombits(erfArg)))
	onePlus := fadd64(math.Float64bits(1.0), erfVal)
	half := math.Float64bits(0.5)
	prod := fmul64(half, fmul64(value64, onePlus))
	got := math.Float32frombits(f64to32(prod))
	fmt.Println("got", got, "ulp", parity.Float32ULPDistance(ref, got))
}
