package metal

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
)

type physicsUnaryFixture struct {
	inputBytes      []byte
	spacingBytes    []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

type physicsTernaryFixture struct {
	firstBytes      []byte
	secondBytes     []byte
	spacingBytes    []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

type physicsFFTFixture struct {
	realInBytes       []byte
	imagInBytes       []byte
	expectedRealBytes []byte
	expectedImagBytes []byte
	expectedReal      []float32
	expectedImag      []float32
}

func physicsUnaryFixtureForTest(
	name string,
	dims []int,
	storageDType dtype.DType,
) physicsUnaryFixture {
	count := productForTest(dims)
	inputValues := physicsValuesForTest(name, count)
	inputBytes := encodeLossValuesAsDType(inputValues, storageDType)
	spacingBytes := encodeLossValuesAsDType([]float32{0.5}, storageDType)
	input := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	spacing := decodeDTypeBytesToFloat32(spacingBytes, storageDType)[0]
	expected := physicsUnaryExpected(name, input, spacing, dims)

	return physicsUnaryFixture{
		inputBytes: inputBytes, spacingBytes: spacingBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func madelungFixtureForTest(count int, storageDType dtype.DType) physicsTernaryFixture {
	densityBytes := encodeLossValuesAsDType(physicsPositiveValues(count, 23), storageDType)
	velocityBytes := encodeLossValuesAsDType(physicsSignedValues(count, 29), storageDType)
	spacingBytes := encodeLossValuesAsDType([]float32{0.5}, storageDType)
	density := decodeDTypeBytesToFloat32(densityBytes, storageDType)
	velocity := decodeDTypeBytesToFloat32(velocityBytes, storageDType)
	spacing := decodeDTypeBytesToFloat32(spacingBytes, storageDType)[0]
	expected := madelungExpected(density, velocity, spacing)

	return physicsTernaryFixture{
		firstBytes: densityBytes, secondBytes: velocityBytes, spacingBytes: spacingBytes,
		expectedBytes: encodeLossValuesAsDType(expected, storageDType), expectedFloat32: expected,
	}
}

func physicsFFTFixtureForTest(
	count int,
	storageDType dtype.DType,
	inverse bool,
) physicsFFTFixture {
	realBytes := encodeLossValuesAsDType(physicsFFTRealValues(count), storageDType)
	imagBytes := encodeLossValuesAsDType(physicsFFTImagValues(count), storageDType)
	realValues := decodeDTypeBytesToFloat32(realBytes, storageDType)
	imagValues := decodeDTypeBytesToFloat32(imagBytes, storageDType)
	expectedReal, expectedImag := fftExpected(realValues, imagValues, inverse)

	return physicsFFTFixture{
		realInBytes: realBytes, imagInBytes: imagBytes,
		expectedRealBytes: encodeLossValuesAsDType(expectedReal, storageDType),
		expectedImagBytes: encodeLossValuesAsDType(expectedImag, storageDType),
		expectedReal:      expectedReal,
		expectedImag:      expectedImag,
	}
}

func physicsUnaryExpected(name string, input []float32, spacing float32, dims []int) []float32 {
	switch name {
	case "laplacian":
		return laplacianExpected(input, spacing, dims)
	case "laplacian4":
		return laplacian4Expected(input, spacing)
	case "grad1d", "divergence1d":
		return gradExpected(input, spacing)
	case "quantum_potential":
		return quantumPotentialExpected(input, spacing)
	default:
		return bohmianVelocityExpected(input, spacing)
	}
}

func physicsValuesForTest(name string, count int) []float32 {
	if name == "quantum_potential" {
		return physicsPositiveValues(count, 17)
	}

	return physicsSignedValues(count, 19)
}

func physicsPositiveValues(count int, salt int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = 0.25 + float32((index*salt+5)%31)/64
	}

	return values
}

func physicsSignedValues(count int, salt int) []float32 {
	values := make([]float32, count)

	for index := range values {
		values[index] = float32((index*salt+7)%37-18) / 16
	}

	return values
}

func physicsFFTRealValues(count int) []float32 {
	values := make([]float32, count)
	if count == 0 {
		return values
	}

	values[0] = 1
	if physicsIsPowerOfTwo(count) {
		physicsFFTPowerOfTwoRealValues(values)
		return values
	}

	if count > 1 {
		values[1] = 0.5
	}
	if count > 3 {
		values[3] = -0.25
	}

	return values
}

func physicsFFTImagValues(count int) []float32 {
	values := make([]float32, count)
	if physicsIsPowerOfTwo(count) {
		physicsFFTPowerOfTwoImagValues(values)
		return values
	}

	if count > 2 {
		values[2] = 0.375
	}
	if count > 5 {
		values[5] = -0.125
	}

	return values
}

func physicsFFTPowerOfTwoRealValues(values []float32) {
	count := len(values)
	if count > 1 {
		values[count/2] = 0.5
	}
	if count > 4 {
		values[count/4] = -0.25
	}
}

func physicsFFTPowerOfTwoImagValues(values []float32) {
	count := len(values)
	if count > 4 {
		values[count/4] = 0.375
	}
	if count > 8 {
		values[count/8] = -0.125
	}
}

func laplacianExpected(input []float32, spacing float32, dims []int) []float32 {
	out := make([]float32, len(input))
	dxSquared := spacing * spacing

	switch len(dims) {
	case 1:
		laplacian1DExpected(input, out, dims[0], dxSquared)
	case 2:
		laplacian2DExpected(input, out, dims[0], dims[1], dxSquared)
	case 3:
		laplacian3DExpected(input, out, dims[0], dims[1], dims[2], dxSquared)
	}

	return out
}

func laplacian1DExpected(input []float32, out []float32, count int, dxSquared float32) {
	for index := range count {
		left := input[(index-1+count)%count]
		right := input[(index+1)%count]
		out[index] = (left - 2*input[index] + right) / dxSquared
	}
}

func laplacian2DExpected(input []float32, out []float32, rows int, cols int, dxSquared float32) {
	for row := range rows {
		for col := range cols {
			index := row*cols + col
			up := input[((row-1+rows)%rows)*cols+col]
			down := input[((row+1)%rows)*cols+col]
			left := input[row*cols+((col-1+cols)%cols)]
			right := input[row*cols+((col+1)%cols)]
			out[index] = (up + down + left + right - 4*input[index]) / dxSquared
		}
	}
}

func laplacian3DExpected(input []float32, out []float32, depth int, rows int, cols int, dxSquared float32) {
	for depthIndex := range depth {
		for row := range rows {
			for col := range cols {
				index := (depthIndex*rows+row)*cols + col
				out[index] = laplacian3DValue(input, depth, rows, cols, depthIndex, row, col) / dxSquared
			}
		}
	}
}

func laplacian3DValue(input []float32, depth int, rows int, cols int, depthIndex int, row int, col int) float32 {
	center := input[(depthIndex*rows+row)*cols+col]
	plane := rows * cols
	dm := input[((depthIndex-1+depth)%depth)*plane+row*cols+col]
	dp := input[((depthIndex+1)%depth)*plane+row*cols+col]
	rm := input[(depthIndex*rows+(row-1+rows)%rows)*cols+col]
	rp := input[(depthIndex*rows+(row+1)%rows)*cols+col]
	cm := input[(depthIndex*rows+row)*cols+(col-1+cols)%cols]
	cp := input[(depthIndex*rows+row)*cols+(col+1)%cols]
	return dm + dp + rm + rp + cm + cp - 6*center
}

func laplacian4Expected(input []float32, spacing float32) []float32 {
	out := make([]float32, len(input))
	denominator := 12 * spacing * spacing
	count := len(input)

	for index := range count {
		out[index] = (-input[(index-2+count)%count] + 16*input[(index-1+count)%count] -
			30*input[index] + 16*input[(index+1)%count] - input[(index+2)%count]) / denominator
	}

	return out
}

func gradExpected(input []float32, spacing float32) []float32 {
	out := make([]float32, len(input))
	count := len(input)

	for index := range count {
		out[index] = (input[(index+1)%count] - input[(index-1+count)%count]) / (2 * spacing)
	}

	return out
}

func quantumPotentialExpected(density []float32, spacing float32) []float32 {
	out := make([]float32, len(density))
	if len(out) == 0 {
		return out
	}

	for index := 1; index < len(density)-1; index++ {
		rho := float64(density[index])
		if rho <= 1.0e-12 {
			continue
		}

		sqrtRho := math.Sqrt(rho)
		sqrtLeft := math.Sqrt(math.Max(1.0e-12, float64(density[index-1])))
		sqrtRight := math.Sqrt(math.Max(1.0e-12, float64(density[index+1])))
		laplacian := (sqrtRight - 2*sqrtRho + sqrtLeft) / float64(spacing*spacing)
		out[index] = float32(-0.5 * laplacian / sqrtRho)
	}

	return out
}

func bohmianVelocityExpected(phase []float32, spacing float32) []float32 {
	out := make([]float32, len(phase))

	for index := 1; index < len(phase)-1; index++ {
		out[index] = (phase[index+1] - phase[index-1]) / (2 * spacing)
	}

	return out
}

func madelungExpected(density []float32, velocity []float32, spacing float32) []float32 {
	out := make([]float32, len(density))

	for index := 1; index < len(density)-1; index++ {
		fluxRight := density[index+1] * velocity[index+1]
		fluxLeft := density[index-1] * velocity[index-1]
		out[index] = (fluxRight - fluxLeft) / (2 * spacing)
	}

	return out
}

func productForTest(dims []int) int {
	product := 1

	for _, dim := range dims {
		product *= dim
	}

	return product
}
