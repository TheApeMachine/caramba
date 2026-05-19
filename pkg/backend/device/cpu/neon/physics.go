package neon

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Quantum-hydrodynamic / PDE physics primitives. Matches the canonical
ops the original substrate's quantum_hydro package exposed, plus the
items its doc.go listed as planned next:

  - laplacian:           2nd-order central-difference Laplacian on
                         rank-1/2/3 grids with periodic BCs.
  - laplacian4:          4th-order central-difference Laplacian.
  - grad1d:              first-derivative central-difference stencil.
  - divergence1d:        divergence stencil ∂F/∂x.
  - fft1d / ifft1d:      complex DFT / inverse DFT (Cooley–Tukey).
  - quantum_potential:   Bohm Q = -ℏ²/(2m) × ∇²√ρ / √ρ.
  - bohmian_velocity:    v = ∇S / m.
  - madelung_continuity: continuity residual ∂ρ/∂t + ∇·(ρv).

All host paths route through Float32Native dispatchers; NEON bodies
live in physics_stencil_f32_neon_arm64.s and physics_f32_dispatch_arm64.go.
*/

func init() {
	Default.Register(Kernel{
		Name: "laplacian",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runLaplacian,
	})

	Default.Register(Kernel{
		Name: "laplacian4",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runLaplacian4,
	})

	Default.Register(Kernel{
		Name: "grad1d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runGrad1D,
	})

	Default.Register(Kernel{
		Name: "divergence1d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runDivergence1D,
	})

	Default.Register(Kernel{
		Name: "fft1d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32, dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runFFT1DDefault,
	})

	Default.Register(Kernel{
		Name: "ifft1d",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32, dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runIFFT1DDefault,
	})

	Default.Register(Kernel{
		Name: "quantum_potential",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runQuantumPotential,
	})

	Default.Register(Kernel{
		Name: "bohmian_velocity",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runBohmianVelocity,
	})

	Default.Register(Kernel{
		Name: "madelung_continuity",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMadelungContinuity,
	})
}

const (
	defaultReducedPlanck = float32(1.0)
	defaultMass          = float32(1.0)
)

/*
runLaplacian computes the 2nd-order central-difference Laplacian
with periodic boundary conditions on a rank-1, rank-2, or rank-3
grid. Args: (input, spacing_scalar, output). Spacing carries one
scalar applied to all spatial axes; per-axis spacing extension lands
when the orchestrator binds anisotropic-spacing configs.

  - 1-D: Δu[i] = (u[i-1] - 2u[i] + u[i+1]) / dx²
  - 2-D: Δu[i,j] = (sum of four neighbours - 4u[i,j]) / dx²
  - 3-D: Δu[i,j,k] = (sum of six neighbours - 6u[i,j,k]) / dx²
*/
func runLaplacian(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	spacing, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(out) != len(input) || len(spacing) < 1 {
		return tensor.ErrShapeMismatch
	}

	dxValue := float64(spacing[0])

	if dxValue <= 0 {
		dxValue = 1.0
	}

	dxSquared := float32(dxValue * dxValue)
	dims := args[0].Shape().Dims()
	invH2 := float32(1.0 / float64(dxSquared))

	switch len(dims) {
	case 1:
		LaplacianFloat32Native(input, out, nil, dims, invH2)
	case 2, 3:
		scratchAxis := dims[0]

		if len(dims) >= 2 && dims[1] > scratchAxis {
			scratchAxis = dims[1]
		}

		scratch := make([]float32, scratchAxis*2)
		LaplacianFloat32Native(input, out, scratch, dims, invH2)
	default:
		return tensor.ErrShapeMismatch
	}

	return nil
}

func laplacian1D(input, out []float32, n int, dxSquared float32) {
	for index := 0; index < n; index++ {
		left := input[(index-1+n)%n]
		right := input[(index+1)%n]
		out[index] = (left - 2*input[index] + right) / dxSquared
	}
}

func laplacian2D(input, out []float32, rows, cols int, dxSquared float32) {
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			center := input[row*cols+col]
			up := input[((row-1+rows)%rows)*cols+col]
			down := input[((row+1)%rows)*cols+col]
			left := input[row*cols+((col-1+cols)%cols)]
			right := input[row*cols+((col+1)%cols)]
			out[row*cols+col] = (up + down + left + right - 4*center) / dxSquared
		}
	}
}

func laplacian3D(input, out []float32, depth, rows, cols int, dxSquared float32) {
	for d := 0; d < depth; d++ {
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				index := (d*rows+row)*cols + col
				center := input[index]
				dm := input[(((d-1+depth)%depth)*rows+row)*cols+col]
				dp := input[(((d+1)%depth)*rows+row)*cols+col]
				rm := input[(d*rows+((row-1+rows)%rows))*cols+col]
				rp := input[(d*rows+((row+1)%rows))*cols+col]
				cm := input[(d*rows+row)*cols+((col-1+cols)%cols)]
				cp := input[(d*rows+row)*cols+((col+1)%cols)]
				out[index] = (dm + dp + rm + rp + cm + cp - 6*center) / dxSquared
			}
		}
	}
}

/*
runLaplacian4 is the 4th-order central-difference Laplacian on a 1-D
periodic grid: Δu[i] = (-u[i-2] + 16u[i-1] - 30u[i] + 16u[i+1] - u[i+2]) / (12 dx²).
*/
func runLaplacian4(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	spacing, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(out) != len(input) || len(spacing) < 1 {
		return tensor.ErrShapeMismatch
	}

	dxValue := float64(spacing[0])

	if dxValue <= 0 {
		dxValue = 1.0
	}

	dxSquared := float32(dxValue * dxValue)
	denominator := 12 * dxSquared
	invDen := float32(1.0 / float64(denominator))

	Laplacian4Float32Native(input, out, invDen)

	return nil
}

/*
runGrad1D computes the first-derivative central-difference stencil
on a 1-D periodic grid: ∂u/∂x[i] = (u[i+1] - u[i-1]) / (2 dx).
*/
func runGrad1D(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	spacing, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(out) != len(input) || len(spacing) < 1 {
		return tensor.ErrShapeMismatch
	}

	dxValue := float64(spacing[0])

	if dxValue <= 0 {
		dxValue = 1.0
	}

	denominator := float32(2 * dxValue)
	invTwoDx := float32(1.0 / float64(denominator))

	Grad1DFloat32Native(input, out, invTwoDx)

	return nil
}

/*
runDivergence1D computes ∂F/∂x on a 1-D periodic grid, the same
central-difference stencil as grad1d but applied to a flux field F.
Conceptually distinct so the orchestrator can route correctly even
when the kernel body coincides.
*/
func runDivergence1D(args ...tensor.Tensor) error {
	return runGrad1D(args...)
}

/*
runFFT1DDefault is the Cooley-Tukey radix-2 DFT for power-of-two
input lengths. Args: (realIn, imagIn, realOut, imagOut). For non
power-of-two sizes the kernel falls back to the naive O(N²) DFT so
correctness is preserved at the cost of speed; production calls
should pad to the next power of two.
*/
func runFFT1DDefault(args ...tensor.Tensor) error {
	return fftDispatch(args, false)
}

func runIFFT1DDefault(args ...tensor.Tensor) error {
	return fftDispatch(args, true)
}

func fftDispatch(args []tensor.Tensor, inverse bool) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	realIn, _ := args[0].Float32Native()
	imagIn, _ := args[1].Float32Native()
	realOut, _ := args[2].Float32Native()
	imagOut, _ := args[3].Float32Native()

	if len(realIn) != len(imagIn) ||
		len(realOut) != len(realIn) || len(imagOut) != len(realIn) {
		return tensor.ErrShapeMismatch
	}

	n := len(realIn)

	if n == 0 {
		return nil
	}

	copy(realOut, realIn)
	copy(imagOut, imagIn)

	if isPowerOfTwo(n) {
		cooleyTukey(realOut, imagOut, inverse)
	} else {
		naiveDFT(realOut, imagOut, inverse)
	}

	if inverse {
		invN := float32(1.0 / float64(n))

		for index := range realOut {
			realOut[index] *= invN
			imagOut[index] *= invN
		}
	}

	return nil
}

func isPowerOfTwo(value int) bool {
	return value > 0 && (value&(value-1)) == 0
}

func cooleyTukey(realOut, imagOut []float32, inverse bool) {
	n := len(realOut)

	// Bit-reversal permutation.
	j := 0

	for i := 1; i < n; i++ {
		bit := n >> 1

		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}

		j ^= bit

		if i < j {
			realOut[i], realOut[j] = realOut[j], realOut[i]
			imagOut[i], imagOut[j] = imagOut[j], imagOut[i]
		}
	}

	sign := -1.0

	if inverse {
		sign = 1.0
	}

	for length := 2; length <= n; length <<= 1 {
		angle := sign * 2 * math.Pi / float64(length)
		wReal := float32(math.Cos(angle))
		wImag := float32(math.Sin(angle))

		for start := 0; start < n; start += length {
			curReal := float32(1)
			curImag := float32(0)
			half := length / 2

			for index := 0; index < half; index++ {
				upper := start + index
				lower := upper + half

				tReal := curReal*realOut[lower] - curImag*imagOut[lower]
				tImag := curReal*imagOut[lower] + curImag*realOut[lower]

				realOut[lower] = realOut[upper] - tReal
				imagOut[lower] = imagOut[upper] - tImag
				realOut[upper] += tReal
				imagOut[upper] += tImag

				newReal := curReal*wReal - curImag*wImag
				newImag := curReal*wImag + curImag*wReal
				curReal, curImag = newReal, newImag
			}
		}
	}
}

func naiveDFT(realOut, imagOut []float32, inverse bool) {
	n := len(realOut)
	sign := -1.0

	if inverse {
		sign = 1.0
	}

	tmpReal := make([]float32, n)
	tmpImag := make([]float32, n)
	copy(tmpReal, realOut)
	copy(tmpImag, imagOut)

	for k := 0; k < n; k++ {
		var sumReal, sumImag float32

		for j := 0; j < n; j++ {
			angle := sign * 2 * math.Pi * float64(k) * float64(j) / float64(n)
			cos := float32(math.Cos(angle))
			sin := float32(math.Sin(angle))
			sumReal += tmpReal[j]*cos - tmpImag[j]*sin
			sumImag += tmpReal[j]*sin + tmpImag[j]*cos
		}

		realOut[k] = sumReal
		imagOut[k] = sumImag
	}
}

/*
runQuantumPotential reads a density ρ[N] and a spacing scalar dx and
writes Q[N] = -ℏ²/(2m) × ∇²√ρ / √ρ. Args: (density, dx_scalar, output).
*/
func runQuantumPotential(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	density, _ := args[0].Float32Native()
	dx, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(out) != len(density) || len(dx) < 1 {
		return tensor.ErrShapeMismatch
	}

	const eps = 1e-12
	_ = eps

	dxValue := float64(dx[0])

	if dxValue <= 0 {
		dxValue = 1.0
	}

	dxSquared := dxValue * dxValue
	invH2 := float32(1.0 / dxSquared)
	scale := float32(-float64(defaultReducedPlanck*defaultReducedPlanck) / (2 * float64(defaultMass)))

	QuantumPotentialFloat32Native(density, out, invH2, scale)

	return nil
}

/*
runBohmianVelocity computes v[i] = (S[i+1] - S[i-1]) / (2 dx) / m.
Args: (phase S, dx_scalar, output v).
*/
func runBohmianVelocity(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	phase, _ := args[0].Float32Native()
	dx, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(out) != len(phase) || len(dx) < 1 {
		return tensor.ErrShapeMismatch
	}

	dxValue := float64(dx[0])

	if dxValue <= 0 {
		dxValue = 1.0
	}

	inverseDoubleDx := float32(1.0 / (2 * dxValue * float64(defaultMass)))

	out[0] = 0
	out[len(out)-1] = 0

	CentralDifferenceInteriorFloat32Native(phase, out, inverseDoubleDx)

	return nil
}

/*
runMadelungContinuity computes the continuity residual ∂ρ/∂t + ∇(ρv).
Args: (density ρ, velocity v, dx_scalar, residual output).
*/
func runMadelungContinuity(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	density, _ := args[0].Float32Native()
	velocity, _ := args[1].Float32Native()
	dx, _ := args[2].Float32Native()
	out, _ := args[3].Float32Native()

	if len(density) != len(velocity) || len(out) != len(density) || len(dx) < 1 {
		return tensor.ErrShapeMismatch
	}

	dxValue := float64(dx[0])

	if dxValue <= 0 {
		dxValue = 1.0
	}

	inverseDoubleDx := float32(1.0 / (2 * dxValue))

	MadelungContinuityFloat32Native(density, velocity, out, inverseDoubleDx)

	return nil
}
