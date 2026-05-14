package cpu

import (
	"fmt"
	"math"

	cpuactivation "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/activation"
	cpumath "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ computetensor.Float64ActivationBackend = (*TensorBackend)(nil)
var _ computetensor.Float64MathBackend = (*TensorBackend)(nil)
var _ computetensor.Float64FusedBackend = (*TensorBackend)(nil)

/*
TensorBackend executes kernels against resident host tensors.
*/
type TensorBackend struct {
	storage *computetensor.HostBackend

	relu    *cpuactivation.ReLU
	gelu    *cpuactivation.Gelu
	tanh    *cpuactivation.Tanh
	sigmoid *cpuactivation.Sigmoid
	swiglu  *cpuactivation.SwiGLU

	add    *cpumath.Add
	mul    *cpumath.Mul
	matmul *cpumath.Matmul
}

/*
NewTensorBackend creates the native Go/SIMD persistent tensor backend.
*/
func NewTensorBackend() *TensorBackend {
	return &TensorBackend{
		storage: computetensor.NewHostBackend(),

		relu:    cpuactivation.NewReLU(),
		gelu:    cpuactivation.NewGelu(),
		tanh:    cpuactivation.NewTanh(),
		sigmoid: cpuactivation.NewSigmoid(),
		swiglu:  cpuactivation.NewSwiGLU(),

		add:    cpumath.NewAdd(),
		mul:    cpumath.NewMul(),
		matmul: cpumath.NewMatmul(),
	}
}

/*
Location identifies host storage ownership.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.Host
}

/*
UploadFloat64 copies host values into resident host storage.
*/
func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.storage.UploadFloat64(shape, values)
}

/*
DownloadFloat64 returns resident host tensor data via HostBackend (zero-copy slice alias).

Independent buffers require CloneFloat64 on the tensor.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	tensor computetensor.Float64Tensor,
) ([]float64, error) {
	return tensorBackend.storage.DownloadFloat64(tensor)
}

/*
Close releases the backend storage owner.
*/
func (tensorBackend *TensorBackend) Close() error {
	return tensorBackend.storage.Close()
}

/*
ReLU applies max(0, x) without leaving the host backend.
*/
func (tensorBackend *TensorBackend) ReLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	values, err := tensorBackend.values(input)

	if err != nil {
		return nil, err
	}

	stateDict := state.NewDict(tensorBackend).
		WithShape(input.Shape().Dims()).
		WithInput(values)
	outputState, err := tensorBackend.relu.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(input.Shape(), outputState.Out)
}

/*
LeakyReLU applies max(alpha*x, x) without leaving the host backend.
*/
func (tensorBackend *TensorBackend) LeakyReLU(
	input computetensor.Float64Tensor, alpha float64,
) (computetensor.Float64Tensor, error) {
	values, err := tensorBackend.values(input)

	if err != nil {
		return nil, err
	}

	operation := cpuactivation.NewLeakyReLU()
	stateDict := state.NewDict(tensorBackend).
		WithShape(input.Shape().Dims()).
		WithInput(values).
		WithAlpha(alpha)
	outputState, err := operation.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(input.Shape(), outputState.Out)
}

/*
GELU applies approximate GELU without leaving the host backend.
*/
func (tensorBackend *TensorBackend) GELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	values, err := tensorBackend.values(input)

	if err != nil {
		return nil, err
	}

	stateDict := state.NewDict(tensorBackend).
		WithShape(input.Shape().Dims()).
		WithInput(values)
	outputState, err := tensorBackend.gelu.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(input.Shape(), outputState.Out)
}

/*
Tanh applies approximate tanh without leaving the host backend.
*/
func (tensorBackend *TensorBackend) Tanh(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	values, err := tensorBackend.values(input)

	if err != nil {
		return nil, err
	}

	stateDict := state.NewDict(tensorBackend).
		WithShape(input.Shape().Dims()).
		WithInput(values)
	outputState, err := tensorBackend.tanh.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(input.Shape(), outputState.Out)
}

/*
Sigmoid applies approximate sigmoid without leaving the host backend.
*/
func (tensorBackend *TensorBackend) Sigmoid(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	values, err := tensorBackend.values(input)

	if err != nil {
		return nil, err
	}

	stateDict := state.NewDict(tensorBackend).
		WithShape(input.Shape().Dims()).
		WithInput(values)
	outputState, err := tensorBackend.sigmoid.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(input.Shape(), outputState.Out)
}

/*
SwiGLU applies gated activation to a tensor whose final dimension is doubled.
*/
func (tensorBackend *TensorBackend) SwiGLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	values, err := tensorBackend.values(input)

	if err != nil {
		return nil, err
	}

	outputShape, err := swigluOutputShape(input.Shape())

	if err != nil {
		return nil, err
	}

	stateDict := state.NewDict(tensorBackend).
		WithShape(input.Shape().Dims()).
		WithInput(values)
	outputState, err := tensorBackend.swiglu.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(outputShape, outputState.Out)
}

/*
Add performs elementwise addition without leaving the host backend.
*/
func (tensorBackend *TensorBackend) Add(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if left == nil || right == nil {
		return nil, fmt.Errorf("cpu tensor: add requires non-nil tensors")
	}

	if !left.Shape().Equal(right.Shape()) {
		return nil, fmt.Errorf("cpu tensor: add shape mismatch")
	}

	leftValues, err := tensorBackend.values(left)

	if err != nil {
		return nil, err
	}

	rightValues, err := tensorBackend.values(right)

	if err != nil {
		return nil, err
	}

	stateDict := state.NewDict(tensorBackend).
		WithShape(left.Shape().Dims()).
		WithInputs(leftValues, rightValues)
	outputState, err := tensorBackend.add.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(left.Shape(), outputState.Out)
}

/*
Mul performs elementwise multiplication without leaving the host backend.
*/
func (tensorBackend *TensorBackend) Mul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if left == nil || right == nil {
		return nil, fmt.Errorf("cpu tensor: mul requires non-nil tensors")
	}

	if !left.Shape().Equal(right.Shape()) {
		return nil, fmt.Errorf("cpu tensor: mul shape mismatch")
	}

	leftValues, err := tensorBackend.values(left)

	if err != nil {
		return nil, err
	}

	rightValues, err := tensorBackend.values(right)

	if err != nil {
		return nil, err
	}

	stateDict := state.NewDict(tensorBackend).
		WithShape(left.Shape().Dims()).
		WithInputs(leftValues, rightValues)
	outputState, err := tensorBackend.mul.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(left.Shape(), outputState.Out)
}

/*
Matmul performs row-major matrix multiplication without leaving the host backend.
*/
func (tensorBackend *TensorBackend) Matmul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if left == nil || right == nil {
		return nil, fmt.Errorf("cpu tensor: matmul requires non-nil tensors")
	}

	leftShape := left.Shape()
	rightShape := right.Shape()
	leftDims := leftShape.Dims()
	rightDims := rightShape.Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 {
		return nil, fmt.Errorf("cpu tensor: matmul requires rank-2 tensors")
	}

	if leftDims[1] != rightDims[0] {
		return nil, fmt.Errorf(
			"cpu tensor: matmul dimension mismatch [%d,%d] x [%d,%d]",
			leftDims[0], leftDims[1], rightDims[0], rightDims[1],
		)
	}

	leftValues, err := tensorBackend.values(left)

	if err != nil {
		return nil, err
	}

	rightValues, err := tensorBackend.values(right)

	if err != nil {
		return nil, err
	}

	outputShape, err := computetensor.NewShape([]int{leftDims[0], rightDims[1]})

	if err != nil {
		return nil, err
	}

	stateDict := state.NewDict(tensorBackend).
		WithShape([]int{leftDims[0], leftDims[1], rightDims[1]}).
		WithInputs(leftValues, rightValues)
	outputState, err := tensorBackend.matmul.Forward(stateDict)

	if err != nil {
		return nil, err
	}

	return tensorBackend.storage.AdoptFloat64(outputShape, outputState.Out)
}

/*
MatmulAdd performs row-major matrix multiplication with a broadcast bias.
*/
func (tensorBackend *TensorBackend) MatmulAdd(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.matmulAdd(left, right, bias, false)
}

/*
MatmulAddGELU performs row-major matrix multiplication, bias, and GELU in one pass.
*/
func (tensorBackend *TensorBackend) MatmulAddGELU(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.matmulAdd(left, right, bias, true)
}

func (tensorBackend *TensorBackend) values(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	if input == nil {
		return nil, fmt.Errorf("cpu tensor: nil input tensor")
	}

	if input.Location() != computetensor.Host {
		return nil, fmt.Errorf("cpu tensor: cannot execute %s tensor", input.Location())
	}

	hostTensor, ok := input.(*computetensor.HostTensor)

	if !ok {
		return nil, fmt.Errorf("cpu tensor: input is not owned by host backend")
	}

	return hostTensor.Float64()
}

func (tensorBackend *TensorBackend) matmulAdd(
	left, right, bias computetensor.Float64Tensor, gelu bool,
) (computetensor.Float64Tensor, error) {
	leftValues, rightValues, biasValues, outputShape, err := tensorBackend.matmulAddInputs(
		left, right, bias,
	)

	if err != nil {
		return nil, err
	}

	dims := outputShape.Dims()
	leftDims := left.Shape().Dims()
	M, K, N := dims[0], leftDims[1], dims[1]
	output := make([]float64, outputShape.Len())
	biasMode := matmulBiasMode(len(biasValues), M, N)

	for row := 0; row < M; row++ {
		for col := 0; col < N; col++ {
			accumulator := biasAt(biasValues, biasMode, row, col, N)

			for index := 0; index < K; index++ {
				accumulator += leftValues[row*K+index] * rightValues[index*N+col]
			}

			if gelu {
				accumulator = geluValue(accumulator)
			}

			output[row*N+col] = accumulator
		}
	}

	return tensorBackend.storage.AdoptFloat64(outputShape, output)
}

func (tensorBackend *TensorBackend) matmulAddInputs(
	left, right, bias computetensor.Float64Tensor,
) ([]float64, []float64, []float64, computetensor.Shape, error) {
	if left == nil || right == nil || bias == nil {
		return nil, nil, nil, computetensor.Shape{},
			fmt.Errorf("cpu tensor: fused matmul requires non-nil tensors")
	}

	leftDims := left.Shape().Dims()
	rightDims := right.Shape().Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf("cpu tensor: fused matmul requires rank-2 tensors")
	}

	if leftDims[1] != rightDims[0] {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf(
			"cpu tensor: fused matmul dimension mismatch [%d,%d] x [%d,%d]",
			leftDims[0], leftDims[1], rightDims[0], rightDims[1],
		)
	}

	M, N := leftDims[0], rightDims[1]
	biasLen := bias.Len()

	if biasLen != N && biasLen != M*N {
		return nil, nil, nil, computetensor.Shape{}, fmt.Errorf(
			"cpu tensor: fused matmul bias length %d must be N=%d or M*N=%d",
			biasLen, N, M*N,
		)
	}

	leftValues, err := tensorBackend.values(left)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	rightValues, err := tensorBackend.values(right)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	biasValues, err := tensorBackend.values(bias)

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	outputShape, err := computetensor.NewShape([]int{M, N})

	if err != nil {
		return nil, nil, nil, computetensor.Shape{}, err
	}

	return leftValues, rightValues, biasValues, outputShape, nil
}

func matmulBiasMode(biasLen, M, N int) int {
	if biasLen == M*N {
		return 1
	}

	return 0
}

func biasAt(bias []float64, mode, row, col, N int) float64 {
	if mode == 1 {
		return bias[row*N+col]
	}

	return bias[col]
}

func geluValue(value float64) float64 {
	cube := value * value * value
	z := 0.7978845608028654 * (value + 0.044715*cube)

	return 0.5 * value * (1 + math.Tanh(z))
}

func swigluOutputShape(shape computetensor.Shape) (computetensor.Shape, error) {
	dimsCopy := append([]int(nil), shape.Dims()...)

	if shape.Len()%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("cpu tensor: swiglu input length must be even")
	}

	if len(dimsCopy) == 0 {
		return computetensor.NewShape([]int{shape.Len() / 2})
	}

	lastIndex := len(dimsCopy) - 1

	if dimsCopy[lastIndex]%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("cpu tensor: swiglu final dimension must be even")
	}

	dimsCopy[lastIndex] /= 2

	return computetensor.NewShape(dimsCopy)
}
