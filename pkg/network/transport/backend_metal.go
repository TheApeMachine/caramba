//go:build darwin && cgo

package transport

import (
	"context"
	"fmt"
	"strings"

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

	activation, err := computemetal.New(computeConfig.Metal.Metallib("activation.metallib"))
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	activeInferenceOps, err := computemetal.NewActiveInferenceOps(
		computeConfig.Metal.Metallib("active_inference.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	attentionOps, err := computemetal.NewAttention(
		computeConfig.Metal.Metallib("attention.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	causalOps, err := computemetal.NewCausalOps(
		computeConfig.Metal.Metallib("causal.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	convolutionOps, err := computemetal.NewConvolutionOps(
		computeConfig.Metal.Metallib("convolution.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	hawkesOps, err := computemetal.NewHawkes(
		computeConfig.Metal.Metallib("hawkes.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	markovBlanketOps, err := computemetal.NewMarkovBlanket(
		computeConfig.Metal.Metallib("markov_blanket.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	maskingOps, err := computemetal.NewMasking(
		computeConfig.Metal.Metallib("masking.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	mathOps, err := computemetal.NewMathOps(computeConfig.Metal.Metallib("math.metallib"))
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	poolingOps, err := computemetal.NewPoolingOps(
		computeConfig.Metal.Metallib("pooling.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	positionalOps, err := computemetal.NewPositional(
		computeConfig.Metal.Metallib("positional.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	predictiveCodingOps, err := computemetal.NewPredictiveCodingOps(
		computeConfig.Metal.Metallib("predictive_coding.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	projectionOps, err := computemetal.NewProjectionOps(
		computeConfig.Metal.Metallib("projection.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	shapeOps, err := computemetal.NewShapeOps(
		computeConfig.Metal.Metallib("shape.metallib"),
	)
	if err != nil {
		_ = tensorBackend.Close()

		return nil, err
	}

	vsaOps, err := computemetal.NewVSAOps(computeConfig.Metal.Metallib("vsa.metallib"))
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

func (backend *MetalStreamBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	switch executor.NormalizeOperation(node.Op) {
	case ir.OpInput:
		return nil, fmt.Errorf("metal tensor: input node %q was not materialized", node.ID)
	case ir.OpAdd:
		return requireMetalInputs(node, inputs, 2, backend.Add)
	case ir.OpMul:
		return requireMetalInputs(node, inputs, 2, backend.Mul)
	case ir.OpMatmul:
		return requireMetalInputs(node, inputs, 2, backend.Matmul)
	case ir.OpReLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ computetensor.Float64Tensor,
		) (computetensor.Float64Tensor, error) {
			return backend.ReLU(input)
		})
	case ir.OpLeakyReLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ computetensor.Float64Tensor,
		) (computetensor.Float64Tensor, error) {
			return backend.LeakyReLU(input, 0.01)
		})
	case ir.OpGELU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ computetensor.Float64Tensor,
		) (computetensor.Float64Tensor, error) {
			return backend.GELU(input)
		})
	case ir.OpTanh:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ computetensor.Float64Tensor,
		) (computetensor.Float64Tensor, error) {
			return backend.Tanh(input)
		})
	case ir.OpSigmoid:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ computetensor.Float64Tensor,
		) (computetensor.Float64Tensor, error) {
			return backend.Sigmoid(input)
		})
	case ir.OpSwiGLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ computetensor.Float64Tensor,
		) (computetensor.Float64Tensor, error) {
			return backend.SwiGLU(input)
		})
	case ir.OpFused:
		if len(inputs) != 3 {
			return nil, fmt.Errorf("metal tensor: Fused node %q requires 3 inputs", node.ID)
		}

		activation, _ := node.Metadata["activation"].(string)
		if strings.EqualFold(activation, string(ir.OpGELU)) {
			return backend.MatmulAddGELU(inputs[0], inputs[1], inputs[2])
		}

		return backend.MatmulAdd(inputs[0], inputs[1], inputs[2])
	default:
		return backend.applyOperation(ctx, node, inputs)
	}
}

func (backend *MetalStreamBackend) applyOperation(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	switch strings.ToLower(string(node.Op)) {
	case "active_inference.belief_update":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.activeInference.BeliefUpdate)
	case "active_inference.expected_free_energy":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.activeInference.ExpectedFreeEnergy)
	case "active_inference.free_energy":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.activeInference.FreeEnergy)
	case "active_inference.precision_weight":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.activeInference.PrecisionWeight)
	case "attention.sdpa", "attention.mqa", "attention.gqa", "attention.sliding_window":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.attention.Forward)
	case "causal.backdoor_adjustment":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.causal.BackdoorAdjustment)
	case "causal.cate":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.causal.CATE)
	case "causal.dag_markov_factorization":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.causal.DAGMarkovFactorization)
	case "causal.do_calculus":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.causal.DoCalculus)
	case "causal.iv_estimate":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.causal.IVEstimate)
	case "convolution.conv1d", "convolution.conv2d", "convolution.conv3d", "convolution.conv_transpose2d":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.convolution.Forward)
	case "hawkes.intensity":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.hawkes.Intensity)
	case "hawkes.kernel_matrix":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.hawkes.KernelMatrix)
	case "hawkes.log_likelihood":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.hawkes.LogLikelihood)
	case "hawkes.simulate":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.hawkes.Simulate)
	case "markov_blanket.flow_active":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.markovBlanket.FlowActive)
	case "markov_blanket.flow_internal":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.markovBlanket.FlowInternal)
	case "markov_blanket.mutual_information":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.markovBlanket.MutualInformation)
	case "markov_blanket.partition":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.markovBlanket.Partition)
	case "masking.apply":
		return executor.RunOperation(ctx, backend, node, inputs, backend.masking.NewApplyMask())
	case "masking.causal":
		return executor.RunOperation(ctx, backend, node, inputs, backend.masking.NewCausalMask())
	case "pooling.adaptive_avg_pool2d", "pooling.adaptive_max_pool2d", "pooling.avg_pool2d", "pooling.max_pool2d":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.pooling.Forward)
	case "positional.alibi", "positional.rope":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.positional.Forward)
	case "predictive_coding.prediction":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.predictiveCoding.Prediction)
	case "predictive_coding.prediction_error":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.predictiveCoding.PredictionError)
	case "predictive_coding.update_representation":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.predictiveCoding.UpdateRepresentation)
	case "predictive_coding.update_weights":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.predictiveCoding.UpdateWeights)
	case "projection.fused_qkv", "projection.linear":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.projection.Forward)
	case "shape.concat", "shape.merge_heads", "shape.reshape", "shape.split", "shape.transpose", "shape.view_as_heads":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.shape.Forward)
	case "vsa.bind":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.vsa.Bind)
	case "vsa.bundle":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.vsa.Bundle)
	case "vsa.similarity":
		return executor.RunErrorOperation(ctx, backend, node, inputs, backend.vsa.Similarity)
	default:
		return nil, fmt.Errorf("metal tensor: unsupported operation %q", node.Op)
	}
}

func requireMetalInputs(
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
	count int,
	apply func(computetensor.Float64Tensor, computetensor.Float64Tensor) (computetensor.Float64Tensor, error),
) (computetensor.Float64Tensor, error) {
	if len(inputs) != count {
		return nil, fmt.Errorf("metal tensor: %s node %q requires %d inputs", node.Op, node.ID, count)
	}

	var second computetensor.Float64Tensor
	if len(inputs) > 1 {
		second = inputs[1]
	}

	return apply(inputs[0], second)
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
