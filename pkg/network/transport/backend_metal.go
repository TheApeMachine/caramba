//go:build darwin && cgo

package transport

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computemetal "github.com/theapemachine/caramba/pkg/backend/compute/metal"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/config"
)

func init() {
	acceleratorStreamBackendFactories = append(
		acceleratorStreamBackendFactories,
		registeredStreamBackendFactory{
			name:    "metal",
			factory: NewMetalStreamBackend,
		},
	)
}

type MetalStreamBackend struct {
	tensors          *computemetal.TensorBackend
	activation       *computemetal.MetalActivation
	activeInference  *computemetal.ActiveInferenceOps
	attention        *computemetal.MetalAttention
	causal           *computemetal.MetalCausalOps
	convolution      *computemetal.ConvolutionOps
	hawkes           *computemetal.MetalHawkes
	markovBlanket    *computemetal.MetalMarkovBlanket
	masking          *computemetal.MetalMasking
	math             *computemetal.MathOps
	pooling          *computemetal.PoolingOps
	positional       *computemetal.MetalPositional
	predictiveCoding *computemetal.MetalPredictiveCodingOps
	projection       *computemetal.ProjectionOps
	shape            *computemetal.MetalShapeOps
	vsa              *computemetal.MetalVSAOps
}

func NewMetalStreamBackend() (StreamComputeBackend, error) {
	computeConfig := config.NewComputeConfig()

	tensorBackend, err := computemetal.NewTensorBackend()
	if err != nil {
		return nil, err
	}

	activation, err := computemetal.New(computeConfig.Metal.ActivationMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	activeInferenceOps, err := computemetal.NewActiveInferenceOps(
		computeConfig.Metal.ActiveInferenceMetallib,
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	attentionOps, err := computemetal.NewAttention(computeConfig.Metal.AttentionMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	causalOps, err := computemetal.NewCausalOps(computeConfig.Metal.CausalMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	convolutionOps, err := computemetal.NewConvolutionOps(
		computeConfig.Metal.ConvolutionMetallib,
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	hawkesOps, err := computemetal.NewHawkes(computeConfig.Metal.HawkesMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	markovBlanketOps, err := computemetal.NewMarkovBlanket(
		computeConfig.Metal.MarkovBlanketMetallib,
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	maskingOps, err := computemetal.NewMasking(computeConfig.Metal.MaskingMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	mathOps, err := computemetal.NewMathOps(computeConfig.Metal.MathMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	poolingOps, err := computemetal.NewPoolingOps(computeConfig.Metal.PoolingMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	positionalOps, err := computemetal.NewPositional(computeConfig.Metal.PositionalMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	predictiveCodingOps, err := computemetal.NewPredictiveCodingOps(
		computeConfig.Metal.PredictiveCodingMetallib,
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	projectionOps, err := computemetal.NewProjectionOps(
		computeConfig.Metal.ProjectionMetallib,
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	shapeOps, err := computemetal.NewShapeOps(computeConfig.Metal.ShapeMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	vsaOps, err := computemetal.NewVSAOps(computeConfig.Metal.VSAMetallib)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	return &MetalStreamBackend{
		tensors:          tensorBackend,
		activation:       activation,
		activeInference:  activeInferenceOps,
		attention:        attentionOps,
		causal:           causalOps,
		convolution:      convolutionOps,
		hawkes:           hawkesOps,
		markovBlanket:    markovBlanketOps,
		masking:          maskingOps,
		math:             mathOps,
		pooling:          poolingOps,
		positional:       positionalOps,
		predictiveCoding: predictiveCodingOps,
		projection:       projectionOps,
		shape:            shapeOps,
		vsa:              vsaOps,
	}, nil
}

func (backend *MetalStreamBackend) Location() computetensor.Location {
	return computetensor.Metal
}

func (backend *MetalStreamBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	return backend.tensors.UploadFloat64(shape, values)
}

func (backend *MetalStreamBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	return backend.tensors.DownloadFloat64(input)
}

func (backend *MetalStreamBackend) Close() error {
	_ = backend.activeInference.Close()
	_ = backend.causal.Close()
	_ = backend.hawkes.Close()
	_ = backend.markovBlanket.Close()
	_ = backend.predictiveCoding.Close()
	_ = backend.vsa.Close()

	return backend.tensors.Close()
}

func (backend *MetalStreamBackend) OperationRegistry() *executor.Registry {
	registry := executor.NewTensorRegistry()
	registry.Register(ir.OpType("active_inference.belief_update"), executor.ErrorOperationHandler(backend.activeInference.BeliefUpdate))
	registry.Register(ir.OpType("active_inference.expected_free_energy"), executor.ErrorOperationHandler(backend.activeInference.ExpectedFreeEnergy))
	registry.Register(ir.OpType("active_inference.free_energy"), executor.ErrorOperationHandler(backend.activeInference.FreeEnergy))
	registry.Register(ir.OpType("active_inference.precision_weight"), executor.ErrorOperationHandler(backend.activeInference.PrecisionWeight))
	registry.Register(ir.OpType("attention.sdpa"), executor.ErrorOperationHandler(backend.attention.Forward))
	registry.Register(ir.OpType("attention.mqa"), executor.ErrorOperationHandler(backend.attention.Forward))
	registry.Register(ir.OpType("attention.gqa"), executor.ErrorOperationHandler(backend.attention.Forward))
	registry.Register(ir.OpType("attention.sliding_window"), executor.ErrorOperationHandler(backend.attention.Forward))
	registry.Register(ir.OpType("causal.backdoor_adjustment"), executor.ErrorOperationHandler(backend.causal.BackdoorAdjustment))
	registry.Register(ir.OpType("causal.cate"), executor.ErrorOperationHandler(backend.causal.CATE))
	registry.Register(ir.OpType("causal.dag_markov_factorization"), executor.ErrorOperationHandler(backend.causal.DAGMarkovFactorization))
	registry.Register(ir.OpType("causal.do_calculus"), executor.ErrorOperationHandler(backend.causal.DoCalculus))
	registry.Register(ir.OpType("causal.iv_estimate"), executor.ErrorOperationHandler(backend.causal.IVEstimate))
	registry.Register(ir.OpType("convolution.conv1d"), executor.ErrorOperationHandler(backend.convolution.Forward))
	registry.Register(ir.OpType("convolution.conv2d"), executor.ErrorOperationHandler(backend.convolution.Forward))
	registry.Register(ir.OpType("convolution.conv3d"), executor.ErrorOperationHandler(backend.convolution.Forward))
	registry.Register(ir.OpType("convolution.conv_transpose2d"), executor.ErrorOperationHandler(backend.convolution.Forward))
	registry.Register(ir.OpType("hawkes.intensity"), executor.ErrorOperationHandler(backend.hawkes.Intensity))
	registry.Register(ir.OpType("hawkes.kernel_matrix"), executor.ErrorOperationHandler(backend.hawkes.KernelMatrix))
	registry.Register(ir.OpType("hawkes.log_likelihood"), executor.ErrorOperationHandler(backend.hawkes.LogLikelihood))
	registry.Register(ir.OpType("hawkes.simulate"), executor.ErrorOperationHandler(backend.hawkes.Simulate))
	registry.Register(ir.OpType("markov_blanket.flow_active"), executor.ErrorOperationHandler(backend.markovBlanket.FlowActive))
	registry.Register(ir.OpType("markov_blanket.flow_internal"), executor.ErrorOperationHandler(backend.markovBlanket.FlowInternal))
	registry.Register(ir.OpType("markov_blanket.mutual_information"), executor.ErrorOperationHandler(backend.markovBlanket.MutualInformation))
	registry.Register(ir.OpType("markov_blanket.partition"), executor.ErrorOperationHandler(backend.markovBlanket.Partition))
	registry.Register(ir.OpType("masking.apply"), executor.OperationHandler(backend.masking.NewApplyMask()))
	registry.Register(ir.OpType("masking.causal"), executor.OperationHandler(backend.masking.NewCausalMask()))
	registry.Register(ir.OpType("pooling.adaptive_avg_pool2d"), executor.ErrorOperationHandler(backend.pooling.Forward))
	registry.Register(ir.OpType("pooling.adaptive_max_pool2d"), executor.ErrorOperationHandler(backend.pooling.Forward))
	registry.Register(ir.OpType("pooling.avg_pool2d"), executor.ErrorOperationHandler(backend.pooling.Forward))
	registry.Register(ir.OpType("pooling.max_pool2d"), executor.ErrorOperationHandler(backend.pooling.Forward))
	registry.Register(ir.OpType("positional.alibi"), executor.ErrorOperationHandler(backend.positional.Forward))
	registry.Register(ir.OpType("positional.rope"), executor.ErrorOperationHandler(backend.positional.Forward))
	registry.Register(ir.OpType("predictive_coding.prediction"), executor.ErrorOperationHandler(backend.predictiveCoding.Prediction))
	registry.Register(ir.OpType("predictive_coding.prediction_error"), executor.ErrorOperationHandler(backend.predictiveCoding.PredictionError))
	registry.Register(ir.OpType("predictive_coding.update_representation"), executor.ErrorOperationHandler(backend.predictiveCoding.UpdateRepresentation))
	registry.Register(ir.OpType("predictive_coding.update_weights"), executor.ErrorOperationHandler(backend.predictiveCoding.UpdateWeights))
	registry.Register(ir.OpType("projection.fused_qkv"), executor.ErrorOperationHandler(backend.projection.Forward))
	registry.Register(ir.OpType("projection.linear"), executor.ErrorOperationHandler(backend.projection.Forward))
	registry.Register(ir.OpType("shape.concat"), executor.ErrorOperationHandler(backend.shape.Forward))
	registry.Register(ir.OpType("shape.merge_heads"), executor.ErrorOperationHandler(backend.shape.Forward))
	registry.Register(ir.OpType("shape.reshape"), executor.ErrorOperationHandler(backend.shape.Forward))
	registry.Register(ir.OpType("shape.split"), executor.ErrorOperationHandler(backend.shape.Forward))
	registry.Register(ir.OpType("shape.transpose"), executor.ErrorOperationHandler(backend.shape.Forward))
	registry.Register(ir.OpType("shape.view_as_heads"), executor.ErrorOperationHandler(backend.shape.Forward))
	registry.Register(ir.OpType("vsa.bind"), executor.ErrorOperationHandler(backend.vsa.Bind))
	registry.Register(ir.OpType("vsa.bundle"), executor.ErrorOperationHandler(backend.vsa.Bundle))
	registry.Register(ir.OpType("vsa.similarity"), executor.ErrorOperationHandler(backend.vsa.Similarity))

	return registry
}

func (backend *MetalStreamBackend) ReLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.activation.ReLUTensor(input)
}

func (backend *MetalStreamBackend) LeakyReLU(
	input computetensor.Float64Tensor, alpha float64,
) (computetensor.Float64Tensor, error) {
	return backend.activation.LeakyReLUTensor(input, alpha)
}

func (backend *MetalStreamBackend) GELU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.activation.GELUTensor(input)
}

func (backend *MetalStreamBackend) Tanh(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.activation.TanhTensor(input)
}

func (backend *MetalStreamBackend) Sigmoid(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.activation.SigmoidTensor(input)
}

func (backend *MetalStreamBackend) SwiGLU(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.activation.SwiGLUTensor(input)
}

func (backend *MetalStreamBackend) Add(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.math.AddTensor(left, right)
}

func (backend *MetalStreamBackend) Mul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.math.MulTensor(left, right)
}

func (backend *MetalStreamBackend) Matmul(
	left, right computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.math.MatmulTensor(left, right)
}

func (backend *MetalStreamBackend) MatmulAdd(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.math.MatmulAddTensor(left, right, bias)
}

func (backend *MetalStreamBackend) MatmulAddGELU(
	left, right, bias computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return backend.math.MatmulAddGELUTensor(left, right, bias)
}
