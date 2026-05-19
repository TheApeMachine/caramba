package kernels

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestRoPEFloat32(t *testing.T) {
	convey.Convey("Given a single token at position 0 with identity rotations", t, func() {
		// At position 0 every θ is 0, so RoPE is identity.
		shape, _ := tensor.NewShape([]int{1, 1, 4})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		copy(inputView, []float32{1, 2, 3, 4})

		err := RunRoPEFloat32(DefaultRoPEConfig(), input, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(outView, convey.ShouldResemble, []float32{1, 2, 3, 4})
	})

	convey.Convey("Given a non-zero position the rotation is non-trivial", t, func() {
		shape, _ := tensor.NewShape([]int{1, 1, 4})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		inputView[0] = 1
		inputView[1] = 0
		inputView[2] = 1
		inputView[3] = 0

		config := DefaultRoPEConfig()
		config.StartPosition = 1

		err := RunRoPEFloat32(config, input, out)
		convey.So(err, convey.ShouldBeNil)

		// For pair 0 with base=10000 and exponent 0: theta = 1.
		// cos(1) ~ 0.5403, sin(1) ~ 0.8415.
		outView, _ := out.Float32Native()
		convey.So(math.Abs(float64(outView[0])-0.5403), convey.ShouldBeLessThan, 1e-3)
		convey.So(math.Abs(float64(outView[1])-0.8415), convey.ShouldBeLessThan, 1e-3)
	})
}

func TestFlashAttentionMatchesBasic(t *testing.T) {
	convey.Convey("Given the same Q/K/V, flash and basic attention agree", t, func() {
		shape, _ := tensor.NewShape([]int{4, 4})
		valueShape, _ := tensor.NewShape([]int{4, 4})

		query, _ := tensor.NewZeroed(shape, dtype.Float32)
		key, _ := tensor.NewZeroed(shape, dtype.Float32)
		value, _ := tensor.NewZeroed(valueShape, dtype.Float32)
		basicOut, _ := tensor.NewZeroed(valueShape, dtype.Float32)
		flashOut, _ := tensor.NewZeroed(valueShape, dtype.Float32)

		queryView, _ := query.Float32Native()
		keyView, _ := key.Float32Native()
		valueView, _ := value.Float32Native()

		for index := range queryView {
			queryView[index] = float32(index%5) * 0.1
			keyView[index] = float32((index*3)%7) * 0.1
			valueView[index] = float32(index%9) * 0.25
		}

		basicKernel, _ := Default.Lookup("attention", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})

		err := basicKernel.Run(query, key, value, basicOut)
		convey.So(err, convey.ShouldBeNil)

		err = RunFlashAttentionFloat32(
			FlashAttentionConfig{BlockSize: 4, Causal: false},
			query, key, value, flashOut,
		)
		convey.So(err, convey.ShouldBeNil)

		basicView, _ := basicOut.Float32Native()
		flashView, _ := flashOut.Float32Native()

		for index := range basicView {
			convey.So(
				math.Abs(float64(basicView[index]-flashView[index])),
				convey.ShouldBeLessThan, 1e-5,
			)
		}
	})
}

func TestAdamWStepWithDecay(t *testing.T) {
	convey.Convey("Given a positive-weight parameter and zero gradient", t, func() {
		shape, _ := tensor.NewShape([]int{2})

		params, _ := tensor.NewZeroed(shape, dtype.Float32)
		grads, _ := tensor.NewZeroed(shape, dtype.Float32)
		firstMoment, _ := tensor.NewZeroed(shape, dtype.Float32)
		secondMoment, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		paramsView, _ := params.Float32Native()
		paramsView[0] = 1.0
		paramsView[1] = -1.0

		config := DefaultAdamWConfig()
		err := AdamWStepFloat32(config, params, grads, firstMoment, secondMoment, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()

		// With zero gradient, the only update comes from weight decay.
		// Positive params decay toward zero; negative params decay
		// toward zero as well (i.e. magnitude shrinks).
		convey.So(outView[0] < paramsView[0], convey.ShouldBeTrue)
		convey.So(outView[1] > paramsView[1], convey.ShouldBeTrue)
	})
}

func TestLionStepDirection(t *testing.T) {
	convey.Convey("Given a positive gradient Lion steps in negative direction", t, func() {
		shape, _ := tensor.NewShape([]int{1})

		params, _ := tensor.NewZeroed(shape, dtype.Float32)
		grads, _ := tensor.NewZeroed(shape, dtype.Float32)
		momentum, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		paramsView, _ := params.Float32Native()
		gradView, _ := grads.Float32Native()

		paramsView[0] = 0
		gradView[0] = 1.0

		err := LionStepFloat32(DefaultLionConfig(), params, grads, momentum, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(outView[0] < 0, convey.ShouldBeTrue)
	})
}

func TestSGDStepDirection(t *testing.T) {
	convey.Convey("Given a positive gradient SGD steps in negative direction", t, func() {
		shape, _ := tensor.NewShape([]int{1})

		params, _ := tensor.NewZeroed(shape, dtype.Float32)
		grads, _ := tensor.NewZeroed(shape, dtype.Float32)
		momentum, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		paramsView, _ := params.Float32Native()
		gradView, _ := grads.Float32Native()

		paramsView[0] = 1.0
		gradView[0] = 1.0

		err := SGDStepFloat32(DefaultSGDConfig(), params, grads, momentum, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(outView[0] < paramsView[0], convey.ShouldBeTrue)
	})
}

func TestInt8DequantRoundTrip(t *testing.T) {
	convey.Convey("Int8 dequant should invert quant under unit scale", t, func() {
		shape, _ := tensor.NewShape([]int{4})

		floats, _ := tensor.NewZeroed(shape, dtype.Float32)
		quantized, _ := tensor.NewZeroed(shape, dtype.Int8)
		dequantized, _ := tensor.NewZeroed(shape, dtype.Float32)

		floatsView, _ := floats.Float32Native()
		copy(floatsView, []float32{0, 1, -1, 50})

		err := QuantInt8Float32(DequantInt8Config{Scale: 1, ZeroPoint: 0}, floats, quantized)
		convey.So(err, convey.ShouldBeNil)

		err = DequantInt8Float32(DequantInt8Config{Scale: 1, ZeroPoint: 0}, quantized, dequantized)
		convey.So(err, convey.ShouldBeNil)

		dequantView, _ := dequantized.Float32Native()
		convey.So(dequantView, convey.ShouldResemble, []float32{0, 1, -1, 50})
	})
}

func TestConv2DIdentityKernel(t *testing.T) {
	convey.Convey("A 1x1 identity kernel should reproduce the input", t, func() {
		inputShape, _ := tensor.NewShape([]int{1, 1, 3, 3})
		weightShape, _ := tensor.NewShape([]int{1, 1, 1, 1})
		biasShape, _ := tensor.NewShape([]int{1})
		outShape, _ := tensor.NewShape([]int{1, 1, 3, 3})

		input, _ := tensor.NewZeroed(inputShape, dtype.Float32)
		weight, _ := tensor.NewZeroed(weightShape, dtype.Float32)
		bias, _ := tensor.NewZeroed(biasShape, dtype.Float32)
		out, _ := tensor.NewZeroed(outShape, dtype.Float32)

		inputView, _ := input.Float32Native()
		copy(inputView, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})

		weightView, _ := weight.Float32Native()
		weightView[0] = 1

		err := Conv2DFloat32(DefaultConv2DConfig(), input, weight, bias, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		convey.So(outView, convey.ShouldResemble, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	})
}

func TestSparseCSRMatMul(t *testing.T) {
	convey.Convey("Sparse CSR matmul should match dense matmul on the same logical values", t, func() {
		backend := tensor.NewHostBackend()
		defer backend.Close()

		// Sparse left: a 3×4 with values at (0,1)=2, (1,2)=3, (2,0)=4.
		shape, _ := tensor.NewShape([]int{3, 4})
		rowPtrShape, _ := tensor.NewShape([]int{4})
		colIdxShape, _ := tensor.NewShape([]int{3})

		rowPtrBytes := make([]byte, 4*4)
		for index, value := range []uint32{0, 1, 2, 3} {
			binary.LittleEndian.PutUint32(rowPtrBytes[index*4:], value)
		}

		colIdxBytes := make([]byte, 3*4)
		for index, value := range []uint32{1, 2, 0} {
			binary.LittleEndian.PutUint32(colIdxBytes[index*4:], value)
		}

		rowPtr, _ := backend.Upload(rowPtrShape, dtype.Int32, rowPtrBytes)
		defer rowPtr.Close()

		colIdx, _ := backend.Upload(colIdxShape, dtype.Int32, colIdxBytes)
		defer colIdx.Close()

		valueBytes := make([]byte, 3*4)
		for index, value := range []float32{2, 3, 4} {
			binary.LittleEndian.PutUint32(valueBytes[index*4:], math.Float32bits(value))
		}

		sparse, _ := backend.UploadSparse(
			shape, dtype.Float32, tensor.LayoutSparseCSR,
			valueBytes,
			[]tensor.SparseIndex{
				{Name: "row_ptr", Data: rowPtr},
				{Name: "col_idx", Data: colIdx},
			},
		)
		defer sparse.Close()

		// Dense right: 4×2 with values 1..8.
		rightShape, _ := tensor.NewShape([]int{4, 2})
		right, _ := tensor.NewZeroed(rightShape, dtype.Float32)
		rightView, _ := right.Float32Native()
		copy(rightView, []float32{1, 2, 3, 4, 5, 6, 7, 8})

		// Output: 3×2.
		outShape, _ := tensor.NewShape([]int{3, 2})
		out, _ := tensor.NewZeroed(outShape, dtype.Float32)

		err := SparseCSRMatMulFloat32(sparse, right, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()
		// Row 0: 2 × [3, 4]      = [6, 8]
		// Row 1: 3 × [5, 6]      = [15, 18]
		// Row 2: 4 × [1, 2]      = [4, 8]
		convey.So(outView, convey.ShouldResemble, []float32{6, 8, 15, 18, 4, 8})
	})
}
