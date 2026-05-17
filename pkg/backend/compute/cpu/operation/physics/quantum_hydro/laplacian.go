package quantum_hydro

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Laplacian computes the discrete Laplacian on a uniform rank-1/2/3 grid
using 2nd-order central differences with periodic boundary conditions.

The struct caches grid spacing and stencil parameters at construction so
Forward only needs to read the input tensor and its OpShape. Spacing is
applied as out = sum_of_neighbour_differences / spacing^2.
*/
type Laplacian struct {
	spacing float64
	order   int
	bc      string
}

/*
NewLaplacian constructs a Laplacian operator with the given uniform grid
spacing. Order and boundary condition default to 2 and "periodic" — the
only values v1 supports. Non-positive spacing falls back to 1.0.
*/
func NewLaplacian(spacing float64) *Laplacian {
	if !(spacing > 0) {
		spacing = 1.0
	}

	return &Laplacian{
		spacing: spacing,
		order:   2,
		bc:      "periodic",
	}
}

/*
Spacing returns the configured uniform grid spacing.
*/
func (laplacian *Laplacian) Spacing() float64 { return laplacian.spacing }

/*
Order returns the configured stencil order. v1 supports order 2 only.
*/
func (laplacian *Laplacian) Order() int { return laplacian.order }

/*
Boundary returns the configured boundary condition. v1 supports "periodic" only.
*/
func (laplacian *Laplacian) Boundary() string { return laplacian.bc }

/*
Forward computes the discrete Laplacian of stateDict.Inputs[0] into
stateDict.Out, preserving shape. The spatial rank is taken from OpShape;
inputs of rank > 3 or rank < 1 are rejected.
*/
func (laplacian *Laplacian) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("stencil.laplacian"); err != nil {
		return nil, err
	}

	if laplacian.order != 2 {
		return nil, fmt.Errorf("stencil.laplacian: only order 2 supported in v1 (got %d)", laplacian.order)
	}

	boundary := laplacian.bc

	if stateDict.Boundary != "" {
		boundary = stateDict.Boundary
	}

	if boundary != "periodic" {
		return nil, fmt.Errorf("stencil.laplacian: only \"periodic\" boundary supported in v1 (got %q)", boundary)
	}

	spacing := laplacian.spacing

	if stateDict.Spacing > 0 {
		spacing = stateDict.Spacing
	}

	input := stateDict.Inputs[0]
	shape := stateDict.OperationShape()

	if err := validateShape(shape, len(input)); err != nil {
		return nil, err
	}

	invH2 := 1.0 / (spacing * spacing)

	laplacianKernel(stateDict.Out, input, shape, invH2)

	return stateDict, nil
}

/*
validateShape checks that the inferred OpShape is rank 1/2/3 and that
the product of its dimensions matches the flat input length.
*/
func validateShape(shape []int, length int) error {
	if len(shape) < 1 || len(shape) > 3 {
		return fmt.Errorf(
			"stencil.laplacian: input rank must be 1, 2, or 3 (got %d)",
			len(shape),
		)
	}

	expected := 1

	for _, dim := range shape {
		if dim < 1 {
			return fmt.Errorf("stencil.laplacian: non-positive dimension in shape %v", shape)
		}

		expected *= dim
	}

	if expected != length {
		return fmt.Errorf(
			"stencil.laplacian: shape %v implies %d elements, input has %d",
			shape, expected, length,
		)
	}

	return nil
}

/*
laplacianScalar is the arch-independent scalar reference implementation.
It is the parity ground truth — every SIMD path must match it within tight
ULP bounds. It also serves as the tail/boundary fallback inside dispatchers.

Indexing is row-major (C order). Periodic wrap is applied per axis.
*/
func laplacianScalar(out, src []float64, shape []int, invH2 float64) {
	switch len(shape) {
	case 1:
		laplacian1DScalar(out, src, shape[0], invH2)
	case 2:
		laplacian2DScalar(out, src, shape[0], shape[1], invH2)
	case 3:
		laplacian3DScalar(out, src, shape[0], shape[1], shape[2], invH2)
	}
}

/*
laplacian1DScalar: out[i] = (src[(i-1+N)%N] - 2*src[i] + src[(i+1)%N]) * invH2.
*/
func laplacian1DScalar(out, src []float64, n int, invH2 float64) {
	if n == 0 {
		return
	}

	if n == 1 {
		// Periodic wrap with a single element: both neighbours are the same
		// element, so the second difference collapses to zero.
		out[0] = 0
		return
	}

	for index := 0; index < n; index++ {
		left := index - 1

		if left < 0 {
			left += n
		}

		right := index + 1

		if right >= n {
			right -= n
		}

		out[index] = (src[left] - 2.0*src[index] + src[right]) * invH2
	}
}

/*
laplacian2DScalar: 5-point stencil with periodic wrap on both axes.
out[i,j] = (src[i-1,j] + src[i+1,j] + src[i,j-1] + src[i,j+1] - 4 src[i,j]) * invH2.
*/
func laplacian2DScalar(out, src []float64, h, w int, invH2 float64) {
	if h == 0 || w == 0 {
		return
	}

	for rowIndex := 0; rowIndex < h; rowIndex++ {
		upRow := rowIndex - 1

		if upRow < 0 {
			upRow += h
		}

		downRow := rowIndex + 1

		if downRow >= h {
			downRow -= h
		}

		rowBase := rowIndex * w
		upBase := upRow * w
		downBase := downRow * w

		for columnIndex := 0; columnIndex < w; columnIndex++ {
			leftColumn := columnIndex - 1

			if leftColumn < 0 {
				leftColumn += w
			}

			rightColumn := columnIndex + 1

			if rightColumn >= w {
				rightColumn -= w
			}

			center := src[rowBase+columnIndex]
			horizontal := src[rowBase+leftColumn] + src[rowBase+rightColumn]
			vertical := src[upBase+columnIndex] + src[downBase+columnIndex]

			out[rowBase+columnIndex] = (horizontal + vertical - 4.0*center) * invH2
		}
	}
}

/*
laplacianKernel dispatches to the rank-specific SIMD body. It is the
universal entry called by Forward on every architecture. The body
functions are arch-agnostic Go that orchestrate scalar boundaries and
delegate the contiguous interior sweep to architecture-specific axis
primitives (axisSet / axisAcc / alignedLen, each defined in its own
arch-tagged source file).
*/
func laplacianKernel(out, src []float64, shape []int, invH2 float64) {
	switch len(shape) {
	case 1:
		laplacianKernel1D(out, src, shape[0], invH2)
	case 2:
		laplacianKernel2D(out, src, shape[0], shape[1], invH2)
	case 3:
		laplacianKernel3D(out, src, shape[0], shape[1], shape[2], invH2)
	}
}

/*
laplacianKernel1D processes the interior with the axis-set SIMD primitive
and handles the two periodic-wrap boundaries scalar. For n < 3 the
periodic wrap collapses; the scalar reference handles that case directly.
*/
func laplacianKernel1D(out, src []float64, n int, invH2 float64) {
	if n < 3 {
		laplacian1DScalar(out, src, n, invH2)

		return
	}

	interior := n - 2
	aligned := alignedLen(interior)

	if aligned > 0 {
		axisSet(
			out[1:1+aligned],
			src[0:aligned],
			src[1:1+aligned],
			src[2:2+aligned],
			invH2,
		)
	}

	for index := 1 + aligned; index <= n-2; index++ {
		out[index] = (src[index-1] + src[index+1] - 2.0*src[index]) * invH2
	}

	out[0] = (src[n-1] + src[1] - 2.0*src[0]) * invH2
	out[n-1] = (src[n-2] + src[0] - 2.0*src[n-1]) * invH2
}

/*
laplacianKernel2D processes the inner-column SIMD segment per row using
axisSet for the horizontal contribution and axisAcc for the vertical
(row-neighbour) contribution. Boundary columns and any unaligned tail
inside the inner sweep are handled scalar.
*/
func laplacianKernel2D(out, src []float64, h, w int, invH2 float64) {
	if h < 1 || w < 1 {
		return
	}

	if w < 3 {
		laplacian2DScalar(out, src, h, w, invH2)

		return
	}

	interior := w - 2
	aligned := alignedLen(interior)

	for rowIndex := 0; rowIndex < h; rowIndex++ {
		upRow := rowIndex - 1

		if upRow < 0 {
			upRow += h
		}

		downRow := rowIndex + 1

		if downRow >= h {
			downRow -= h
		}

		rowBase := rowIndex * w
		upBase := upRow * w
		downBase := downRow * w

		if aligned > 0 {
			axisSet(
				out[rowBase+1:rowBase+1+aligned],
				src[rowBase:rowBase+aligned],
				src[rowBase+1:rowBase+1+aligned],
				src[rowBase+2:rowBase+2+aligned],
				invH2,
			)
			axisAcc(
				out[rowBase+1:rowBase+1+aligned],
				src[upBase+1:upBase+1+aligned],
				src[rowBase+1:rowBase+1+aligned],
				src[downBase+1:downBase+1+aligned],
				invH2,
			)
		}

		for columnIndex := 1 + aligned; columnIndex <= w-2; columnIndex++ {
			center := src[rowBase+columnIndex]
			horizontal := src[rowBase+columnIndex-1] + src[rowBase+columnIndex+1]
			vertical := src[upBase+columnIndex] + src[downBase+columnIndex]

			out[rowBase+columnIndex] = (horizontal + vertical - 4.0*center) * invH2
		}

		laplacian2DBoundaryColumn(out, src, rowIndex, upRow, downRow, 0, w, invH2)
		laplacian2DBoundaryColumn(out, src, rowIndex, upRow, downRow, w-1, w, invH2)
	}
}

/*
laplacian2DBoundaryColumn writes out[rowIndex, columnIndex] using full
periodic-wrap addressing on both axes. Used for the j=0 and j=W-1 columns
where the in-row neighbours wrap around.
*/
func laplacian2DBoundaryColumn(
	out, src []float64,
	rowIndex, upRow, downRow, columnIndex, w int,
	invH2 float64,
) {
	leftColumn := columnIndex - 1

	if leftColumn < 0 {
		leftColumn += w
	}

	rightColumn := columnIndex + 1

	if rightColumn >= w {
		rightColumn -= w
	}

	rowBase := rowIndex * w
	center := src[rowBase+columnIndex]
	horizontal := src[rowBase+leftColumn] + src[rowBase+rightColumn]
	vertical := src[upRow*w+columnIndex] + src[downRow*w+columnIndex]

	out[rowBase+columnIndex] = (horizontal + vertical - 4.0*center) * invH2
}

/*
laplacianKernel3D processes the inner-column SIMD segment per (depth, row)
plane line, then handles boundary columns and depth/row periodic wraps
scalar. Each accumulate call contributes one pair of axis neighbours.
*/
func laplacianKernel3D(out, src []float64, d, h, w int, invH2 float64) {
	if d < 1 || h < 1 || w < 1 {
		return
	}

	if w < 3 {
		laplacian3DScalar(out, src, d, h, w, invH2)

		return
	}

	hw := h * w
	interior := w - 2
	aligned := alignedLen(interior)

	for depthIndex := 0; depthIndex < d; depthIndex++ {
		frontDepth := depthIndex - 1

		if frontDepth < 0 {
			frontDepth += d
		}

		backDepth := depthIndex + 1

		if backDepth >= d {
			backDepth -= d
		}

		depthBase := depthIndex * hw
		frontBase := frontDepth * hw
		backBase := backDepth * hw

		for rowIndex := 0; rowIndex < h; rowIndex++ {
			upRow := rowIndex - 1

			if upRow < 0 {
				upRow += h
			}

			downRow := rowIndex + 1

			if downRow >= h {
				downRow -= h
			}

			rowOffset := rowIndex * w
			upOffset := upRow * w
			downOffset := downRow * w

			centerStart := depthBase + rowOffset
			upStart := depthBase + upOffset
			downStart := depthBase + downOffset
			frontStart := frontBase + rowOffset
			backStart := backBase + rowOffset

			if aligned > 0 {
				axisSet(
					out[centerStart+1:centerStart+1+aligned],
					src[centerStart:centerStart+aligned],
					src[centerStart+1:centerStart+1+aligned],
					src[centerStart+2:centerStart+2+aligned],
					invH2,
				)
				axisAcc(
					out[centerStart+1:centerStart+1+aligned],
					src[upStart+1:upStart+1+aligned],
					src[centerStart+1:centerStart+1+aligned],
					src[downStart+1:downStart+1+aligned],
					invH2,
				)
				axisAcc(
					out[centerStart+1:centerStart+1+aligned],
					src[frontStart+1:frontStart+1+aligned],
					src[centerStart+1:centerStart+1+aligned],
					src[backStart+1:backStart+1+aligned],
					invH2,
				)
			}

			for columnIndex := 1 + aligned; columnIndex <= w-2; columnIndex++ {
				cellIndex := centerStart + columnIndex
				center := src[cellIndex]

				horizontal := src[cellIndex-1] + src[cellIndex+1]
				vertical := src[upStart+columnIndex] + src[downStart+columnIndex]
				transverse := src[frontStart+columnIndex] + src[backStart+columnIndex]

				out[cellIndex] = (horizontal + vertical + transverse - 6.0*center) * invH2
			}

			laplacian3DBoundaryColumn(out, src, depthIndex, frontDepth, backDepth, rowIndex, upRow, downRow, 0, h, w, invH2)
			laplacian3DBoundaryColumn(out, src, depthIndex, frontDepth, backDepth, rowIndex, upRow, downRow, w-1, h, w, invH2)
		}
	}
}

/*
laplacian3DBoundaryColumn writes one boundary cell for the 3D case using
full periodic-wrap addressing on all three axes.
*/
func laplacian3DBoundaryColumn(
	out, src []float64,
	depthIndex, frontDepth, backDepth, rowIndex, upRow, downRow, columnIndex, h, w int,
	invH2 float64,
) {
	leftColumn := columnIndex - 1

	if leftColumn < 0 {
		leftColumn += w
	}

	rightColumn := columnIndex + 1

	if rightColumn >= w {
		rightColumn -= w
	}

	hw := h * w
	depthBase := depthIndex * hw
	rowOffset := rowIndex * w
	cellIndex := depthBase + rowOffset + columnIndex

	center := src[cellIndex]
	horizontal := src[depthBase+rowOffset+leftColumn] + src[depthBase+rowOffset+rightColumn]
	vertical := src[depthBase+upRow*w+columnIndex] + src[depthBase+downRow*w+columnIndex]
	transverse := src[frontDepth*hw+rowOffset+columnIndex] + src[backDepth*hw+rowOffset+columnIndex]

	out[cellIndex] = (horizontal + vertical + transverse - 6.0*center) * invH2
}

/*
laplacian3DScalar: 7-point stencil with periodic wrap on all axes.
out[k,i,j] = (sum of 6 axis neighbours - 6 src[k,i,j]) * invH2.
*/
func laplacian3DScalar(out, src []float64, d, h, w int, invH2 float64) {
	if d == 0 || h == 0 || w == 0 {
		return
	}

	hw := h * w

	for depthIndex := 0; depthIndex < d; depthIndex++ {
		frontDepth := depthIndex - 1

		if frontDepth < 0 {
			frontDepth += d
		}

		backDepth := depthIndex + 1

		if backDepth >= d {
			backDepth -= d
		}

		depthBase := depthIndex * hw
		frontBase := frontDepth * hw
		backBase := backDepth * hw

		for rowIndex := 0; rowIndex < h; rowIndex++ {
			upRow := rowIndex - 1

			if upRow < 0 {
				upRow += h
			}

			downRow := rowIndex + 1

			if downRow >= h {
				downRow -= h
			}

			rowOffset := rowIndex * w
			upOffset := upRow * w
			downOffset := downRow * w

			for columnIndex := 0; columnIndex < w; columnIndex++ {
				leftColumn := columnIndex - 1

				if leftColumn < 0 {
					leftColumn += w
				}

				rightColumn := columnIndex + 1

				if rightColumn >= w {
					rightColumn -= w
				}

				cellIndex := depthBase + rowOffset + columnIndex
				center := src[cellIndex]

				horizontal := src[depthBase+rowOffset+leftColumn] +
					src[depthBase+rowOffset+rightColumn]
				vertical := src[depthBase+upOffset+columnIndex] +
					src[depthBase+downOffset+columnIndex]
				transverse := src[frontBase+rowOffset+columnIndex] +
					src[backBase+rowOffset+columnIndex]

				out[cellIndex] = (horizontal + vertical + transverse - 6.0*center) * invH2
			}
		}
	}
}
