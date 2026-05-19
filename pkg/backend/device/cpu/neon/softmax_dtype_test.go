package neon

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestSoftmaxFloat16AndBFloat16(t *testing.T) {
	for _, storageDType := range []dtype.DType{dtype.Float16, dtype.BFloat16} {
		storageDType := storageDType

		t.Run(storageDType.Name(), func(t *testing.T) {
			for _, n := range parityNs {
				n := n

				t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
					convey.Convey("Output should match the dtype softmax reference", t, func() {
						assertSoftmaxDTypeReference(t, storageDType, n)
					})
				})
			}
		})
	}
}

func assertSoftmaxDTypeReference(
	t testing.TB,
	storageDType dtype.DType,
	n int,
) {
	t.Helper()

	kernel, ok := Default.Lookup("softmax", Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	})
	convey.So(ok, convey.ShouldBeTrue)

	shape, _ := tensor.NewShape([]int{1, n})
	input, _ := tensor.NewZeroed(shape, storageDType)
	out, _ := tensor.NewZeroed(shape, storageDType)

	fillSoftmaxDTypeInput(input, storageDType)
	expected := expectedSoftmaxDTypeBits(input, storageDType)

	err := kernel.Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertSoftmaxDTypeBits(t, out, storageDType, expected, 1)
}

func fillSoftmaxDTypeInput(input tensor.Tensor, storageDType dtype.DType) {
	switch storageDType {
	case dtype.Float16:
		view, _ := input.Float16Native()
		for index := range view {
			view[index] = dtype.Fromfloat32(softmaxDTypeInputValue(index))
		}
	case dtype.BFloat16:
		view, _ := input.BFloat16Native()
		for index := range view {
			view[index] = dtype.NewBfloat16FromFloat32(softmaxDTypeInputValue(index))
		}
	}
}

func softmaxDTypeInputValue(index int) float32 {
	return float32(index%37-18) / 8
}

func expectedSoftmaxDTypeBits(input tensor.Tensor, storageDType dtype.DType) []uint16 {
	values := softmaxDTypeStoredValues(input, storageDType)
	expected := expectedSoftmaxFloat32(values)
	return encodeSoftmaxExpectedBits(expected, storageDType)
}

func softmaxDTypeStoredValues(input tensor.Tensor, storageDType dtype.DType) []float32 {
	switch storageDType {
	case dtype.Float16:
		view, _ := input.Float16Native()
		values := make([]float32, len(view))
		for index, value := range view {
			values[index] = value.Float32()
		}

		return values
	case dtype.BFloat16:
		view, _ := input.BFloat16Native()
		values := make([]float32, len(view))
		for index := range view {
			values[index] = (&view[index]).Float32()
		}

		return values
	}

	return nil
}

func expectedSoftmaxFloat32(input []float32) []float32 {
	out := make([]float32, len(input))
	maximum := input[0]

	for _, value := range input[1:] {
		if value > maximum {
			maximum = value
		}
	}

	var sum float32

	for index, value := range input {
		shifted := float32(math.Exp(float64(value - maximum)))
		out[index] = shifted
		sum += shifted
	}

	for index := range out {
		out[index] /= sum
	}

	return out
}

func encodeSoftmaxExpectedBits(values []float32, storageDType dtype.DType) []uint16 {
	out := make([]uint16, len(values))

	switch storageDType {
	case dtype.Float16:
		for index, value := range values {
			out[index] = dtype.Fromfloat32(value).Bits()
		}
	case dtype.BFloat16:
		for index, value := range values {
			converted := dtype.NewBfloat16FromFloat32(value)
			out[index] = (&converted).Bits()
		}
	}

	return out
}

func assertSoftmaxDTypeBits(
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

func softmaxDTypeOutputBits(out tensor.Tensor, storageDType dtype.DType) []uint16 {
	switch storageDType {
	case dtype.Float16:
		view, _ := out.Float16Native()
		values := make([]uint16, len(view))
		for index, value := range view {
			values[index] = value.Bits()
		}

		return values
	case dtype.BFloat16:
		view, _ := out.BFloat16Native()
		values := make([]uint16, len(view))
		for index := range view {
			values[index] = (&view[index]).Bits()
		}

		return values
	}

	return nil
}

func softmaxUint16Distance(actual uint16, expected uint16) uint32 {
	if actual > expected {
		return uint32(actual - expected)
	}

	return uint32(expected - actual)
}
