package neon

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestLayerNormFloat16AndBFloat16(t *testing.T) {
	for _, storageDType := range []dtype.DType{dtype.Float16, dtype.BFloat16} {
		storageDType := storageDType

		t.Run(storageDType.Name(), func(t *testing.T) {
			for _, n := range parityNs {
				n := n

				t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
					convey.Convey("Output should match the dtype layernorm reference", t, func() {
						assertLayerNormDTypeReference(t, storageDType, n)
					})
				})
			}
		})
	}
}

func TestRMSNormFloat16AndBFloat16(t *testing.T) {
	for _, storageDType := range []dtype.DType{dtype.Float16, dtype.BFloat16} {
		storageDType := storageDType

		t.Run(storageDType.Name(), func(t *testing.T) {
			for _, n := range parityNs {
				n := n

				t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
					convey.Convey("Output should match the dtype rmsnorm reference", t, func() {
						assertRMSNormDTypeReference(t, storageDType, n)
					})
				})
			}
		})
	}
}

func assertLayerNormDTypeReference(t testing.TB, storageDType dtype.DType, n int) {
	t.Helper()

	kernel := lookupHostLayerNormDTypeKernel(t, storageDType)
	shape, _ := tensor.NewShape([]int{1, n})
	paramShape, _ := tensor.NewShape([]int{n})
	input, _ := tensor.NewZeroed(shape, storageDType)
	scale, _ := tensor.NewZeroed(paramShape, storageDType)
	bias, _ := tensor.NewZeroed(paramShape, storageDType)
	out, _ := tensor.NewZeroed(shape, storageDType)

	fillNormDTypeInput(input, scale, bias, storageDType)
	expected := expectedLayerNormDTypeBits(input, scale, bias, storageDType)

	err := kernel.Run(input, scale, bias, out)
	convey.So(err, convey.ShouldBeNil)
	assertNormDTypeBits(t, out, storageDType, expected, 1)
}

func assertRMSNormDTypeReference(t testing.TB, storageDType dtype.DType, n int) {
	t.Helper()

	kernel := lookupHostRMSNormDTypeKernel(t, storageDType)
	shape, _ := tensor.NewShape([]int{1, n})
	paramShape, _ := tensor.NewShape([]int{n})
	input, _ := tensor.NewZeroed(shape, storageDType)
	scale, _ := tensor.NewZeroed(paramShape, storageDType)
	out, _ := tensor.NewZeroed(shape, storageDType)

	fillNormDTypeInput(input, scale, nil, storageDType)
	expected := expectedRMSNormDTypeBits(input, scale, storageDType)

	err := kernel.Run(input, scale, out)
	convey.So(err, convey.ShouldBeNil)
	assertNormDTypeBits(t, out, storageDType, expected, 1)
}

func lookupHostLayerNormDTypeKernel(t testing.TB, storageDType dtype.DType) Kernel {
	t.Helper()

	kernel, ok := Default.LookupLocation("layernorm", Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Host)
	convey.So(ok, convey.ShouldBeTrue)
	return kernel
}

func lookupHostRMSNormDTypeKernel(t testing.TB, storageDType dtype.DType) Kernel {
	t.Helper()

	kernel, ok := Default.LookupLocation("rmsnorm", Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Host)
	convey.So(ok, convey.ShouldBeTrue)
	return kernel
}

func fillNormDTypeInput(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	storageDType dtype.DType,
) {
	switch storageDType {
	case dtype.Float16:
		fillNormFloat16Input(input, scale, bias)
	case dtype.BFloat16:
		fillNormBFloat16Input(input, scale, bias)
	}
}

func fillNormFloat16Input(input tensor.Tensor, scale tensor.Tensor, bias tensor.Tensor) {
	inputView, _ := input.Float16Native()
	scaleView, _ := scale.Float16Native()

	for index := range inputView {
		inputView[index] = dtype.Fromfloat32(normDTypeInputValue(index))
	}

	for index := range scaleView {
		scaleView[index] = dtype.Fromfloat32(normDTypeScaleValue(index))
	}

	if bias == nil {
		return
	}

	biasView, _ := bias.Float16Native()
	for index := range biasView {
		biasView[index] = dtype.Fromfloat32(normDTypeBiasValue(index))
	}
}

func fillNormBFloat16Input(input tensor.Tensor, scale tensor.Tensor, bias tensor.Tensor) {
	inputView, _ := input.BFloat16Native()
	scaleView, _ := scale.BFloat16Native()

	for index := range inputView {
		inputView[index] = dtype.NewBfloat16FromFloat32(normDTypeInputValue(index))
	}

	for index := range scaleView {
		scaleView[index] = dtype.NewBfloat16FromFloat32(normDTypeScaleValue(index))
	}

	if bias == nil {
		return
	}

	biasView, _ := bias.BFloat16Native()
	for index := range biasView {
		biasView[index] = dtype.NewBfloat16FromFloat32(normDTypeBiasValue(index))
	}
}

func normDTypeInputValue(index int) float32 {
	return float32(index%41-20) / 16
}

func normDTypeScaleValue(index int) float32 {
	return 1 + float32(index%17-8)/64
}

func normDTypeBiasValue(index int) float32 {
	return float32(index%19-9) / 128
}

func expectedLayerNormDTypeBits(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	storageDType dtype.DType,
) []uint16 {
	inputValues := normDTypeStoredValues(input, storageDType)
	scaleValues := normDTypeStoredValues(scale, storageDType)
	biasValues := normDTypeStoredValues(bias, storageDType)
	expected := expectedLayerNormFloat32(inputValues, scaleValues, biasValues)
	return encodeNormExpectedBits(expected, storageDType)
}

func expectedRMSNormDTypeBits(
	input tensor.Tensor,
	scale tensor.Tensor,
	storageDType dtype.DType,
) []uint16 {
	inputValues := normDTypeStoredValues(input, storageDType)
	scaleValues := normDTypeStoredValues(scale, storageDType)
	expected := expectedRMSNormFloat32(inputValues, scaleValues)
	return encodeNormExpectedBits(expected, storageDType)
}

func normDTypeStoredValues(input tensor.Tensor, storageDType dtype.DType) []float32 {
	if storageDType == dtype.Float16 {
		return normFloat16StoredValues(input)
	}

	return normBFloat16StoredValues(input)
}

func normFloat16StoredValues(input tensor.Tensor) []float32 {
	view, _ := input.Float16Native()
	values := make([]float32, len(view))

	for index, value := range view {
		values[index] = value.Float32()
	}

	return values
}

func normBFloat16StoredValues(input tensor.Tensor) []float32 {
	view, _ := input.BFloat16Native()
	values := make([]float32, len(view))

	for index := range view {
		values[index] = (&view[index]).Float32()
	}

	return values
}

func expectedLayerNormFloat32(input []float32, scale []float32, bias []float32) []float32 {
	mean := normMean(input)
	variance := normVariance(input, mean)
	invStdDev := 1 / float32(math.Sqrt(float64(variance+layerNormEpsilon)))
	out := make([]float32, len(input))

	for index, value := range input {
		out[index] = (value-mean)*invStdDev*scale[index] + bias[index]
	}

	return out
}

func expectedRMSNormFloat32(input []float32, scale []float32) []float32 {
	meanSquare := normMeanSquare(input)
	invRMS := 1 / float32(math.Sqrt(float64(meanSquare+rmsNormEpsilon)))
	out := make([]float32, len(input))

	for index, value := range input {
		out[index] = value * invRMS * scale[index]
	}

	return out
}

func normMean(input []float32) float32 {
	var sum float32

	for _, value := range input {
		sum += value
	}

	return sum / float32(len(input))
}

func normVariance(input []float32, mean float32) float32 {
	var variance float32

	for _, value := range input {
		delta := value - mean
		variance += delta * delta
	}

	return variance / float32(len(input))
}

func normMeanSquare(input []float32) float32 {
	var meanSquare float32

	for _, value := range input {
		meanSquare += value * value
	}

	return meanSquare / float32(len(input))
}

func encodeNormExpectedBits(values []float32, storageDType dtype.DType) []uint16 {
	out := make([]uint16, len(values))

	for index, value := range values {
		out[index] = normDTypeBits(value, storageDType)
	}

	return out
}

func normDTypeBits(value float32, storageDType dtype.DType) uint16 {
	if storageDType == dtype.Float16 {
		return dtype.Fromfloat32(value).Bits()
	}

	converted := dtype.NewBfloat16FromFloat32(value)
	return (&converted).Bits()
}

func assertNormDTypeBits(
	t testing.TB,
	out tensor.Tensor,
	storageDType dtype.DType,
	expected []uint16,
	maxULP uint32,
) {
	t.Helper()

	actual := softmaxDTypeOutputBits(out, storageDType)
	for index, actualBits := range actual {
		distance := softmaxUint16Distance(actualBits, expected[index])
		if distance <= maxULP {
			continue
		}

		t.Fatalf(
			"dtype bit mismatch at element %d: got %04x, want %04x, distance %d > %d",
			index,
			actualBits,
			expected[index],
			distance,
			maxULP,
		)
	}
}
