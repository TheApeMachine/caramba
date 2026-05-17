package dispatch

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type OperationRegistry interface {
	ReLU(*state.Dict) (state.Operation, error)
	LeakyReLU(*state.Dict) (state.Operation, error)
	GELU(*state.Dict) (state.Operation, error)
	Tanh(*state.Dict) (state.Operation, error)
	Sigmoid(*state.Dict) (state.Operation, error)
	SwiGLU(*state.Dict) (state.Operation, error)
	Swish(*state.Dict) (state.Operation, error)
	SELU(*state.Dict) (state.Operation, error)
	SDPA(*state.Dict) (state.Operation, error)
	MQA(*state.Dict) (state.Operation, error)
	GQA(*state.Dict) (state.Operation, error)
	SlidingWindowAttention(*state.Dict) (state.Operation, error)
	ApplyMask(*state.Dict) (state.Operation, error)
	CausalMask(*state.Dict) (state.Operation, error)
	Add(*state.Dict) (state.Operation, error)
	Mul(*state.Dict) (state.Operation, error)
	Matmul(*state.Dict) (state.Operation, error)
	Exp(*state.Dict) (state.Operation, error)
	Sin(*state.Dict) (state.Operation, error)
	Cos(*state.Dict) (state.Operation, error)
	Log(*state.Dict) (state.Operation, error)
	LogSumExp(*state.Dict) (state.Operation, error)
	Softmax(*state.Dict) (state.Operation, error)
	Outer(*state.Dict) (state.Operation, error)
	Sign(*state.Dict) (state.Operation, error)
	InvSqrtDimScale(*state.Dict) (state.Operation, error)
	Dropout(*state.Dict) (state.Operation, error)
	RMSNorm(*state.Dict) (state.Operation, error)
	LayerNorm(*state.Dict) (state.Operation, error)
	GroupNorm(*state.Dict) (state.Operation, error)
	Reshape(*state.Dict) (state.Operation, error)
	Transpose(*state.Dict) (state.Operation, error)
	Concat(*state.Dict) (state.Operation, error)
	Split(*state.Dict) (state.Operation, error)
	UpsampleNearest2D(*state.Dict) (state.Operation, error)
	ViewAsHeads(*state.Dict) (state.Operation, error)
	MergeHeads(*state.Dict) (state.Operation, error)
	LastToken(*state.Dict) (state.Operation, error)
	Slice(*state.Dict) (state.Operation, error)
	RoPE(*state.Dict) (state.Operation, error)
	ALiBi(*state.Dict) (state.Operation, error)
	TokenEmbedding(*state.Dict) (state.Operation, error)
	Conv1D(*state.Dict) (state.Operation, error)
	Conv2D(*state.Dict) (state.Operation, error)
	Conv3D(*state.Dict) (state.Operation, error)
	ConvTranspose2D(*state.Dict) (state.Operation, error)
	MaxPool2D(*state.Dict) (state.Operation, error)
	AvgPool2D(*state.Dict) (state.Operation, error)
	AdaptiveAvgPool2D(*state.Dict) (state.Operation, error)
	AdaptiveMaxPool2D(*state.Dict) (state.Operation, error)
	Linear(*state.Dict) (state.Operation, error)
	FusedQKV(*state.Dict) (state.Operation, error)
	HawkesIntensity(*state.Dict) (state.Operation, error)
	HawkesKernelMatrix(*state.Dict) (state.Operation, error)
	HawkesLogLikelihood(*state.Dict) (state.Operation, error)
	HawkesSimulate(*state.Dict) (state.Operation, error)
	VSABind(*state.Dict) (state.Operation, error)
	VSABundle(*state.Dict) (state.Operation, error)
	VSASimilarity(*state.Dict) (state.Operation, error)
	VSAPermute(*state.Dict) (state.Operation, error)
	VSAInversePermute(*state.Dict) (state.Operation, error)
	BeliefUpdate(*state.Dict) (state.Operation, error)
	ExpectedFreeEnergy(*state.Dict) (state.Operation, error)
	FreeEnergy(*state.Dict) (state.Operation, error)
	PrecisionWeight(*state.Dict) (state.Operation, error)
	Prediction(*state.Dict) (state.Operation, error)
	PredictionError(*state.Dict) (state.Operation, error)
	UpdateRepresentation(*state.Dict) (state.Operation, error)
	UpdateWeights(*state.Dict) (state.Operation, error)
	FlowActive(*state.Dict) (state.Operation, error)
	FlowInternal(*state.Dict) (state.Operation, error)
	MutualInformation(*state.Dict) (state.Operation, error)
	Partition(*state.Dict) (state.Operation, error)
	BackdoorAdjustment(*state.Dict) (state.Operation, error)
	CATE(*state.Dict) (state.Operation, error)
	Counterfactual(*state.Dict) (state.Operation, error)
	DAGMarkovFactorization(*state.Dict) (state.Operation, error)
	DoCalculus(*state.Dict) (state.Operation, error)
	FrontdoorAdjustment(*state.Dict) (state.Operation, error)
	IVEstimate(*state.Dict) (state.Operation, error)
	MSELoss(*state.Dict) (state.Operation, error)
	CrossEntropyLoss(*state.Dict) (state.Operation, error)
	MSEGrad(*state.Dict) (state.Operation, error)
	CrossEntropyGrad(*state.Dict) (state.Operation, error)
	Accuracy(*state.Dict) (state.Operation, error)
	Perplexity(*state.Dict) (state.Operation, error)
	F1(*state.Dict) (state.Operation, error)
	Graft(*state.Dict) (state.Operation, error)
	Freeze(*state.Dict) (state.Operation, error)
}

type OptimizerRegistry interface {
	Adam(*state.Dict) (state.Optimizer, error)
	AdamW(*state.Dict) (state.Optimizer, error)
	AdaMax(*state.Dict) (state.Optimizer, error)
	SGD(*state.Dict) (state.Optimizer, error)
	Lion(*state.Dict) (state.Optimizer, error)
	RMSProp(*state.Dict) (state.Optimizer, error)
	Hebbian(*state.Dict) (state.Optimizer, error)
	Lars(*state.Dict) (state.Optimizer, error)
	Lamb(*state.Dict) (state.Optimizer, error)
	AdaGrad(*state.Dict) (state.Optimizer, error)
	AdaDelta(*state.Dict) (state.Optimizer, error)
	LBFGS(*state.Dict) (state.Optimizer, error)
}

func SupportedIDSet() map[ir.OpType]bool {
	supported := make(map[ir.OpType]bool, len(ir.RequiredOperationIDs()))

	for _, operationID := range ir.RequiredOperationIDs() {
		supported[operationID] = true
	}

	return supported
}

func RunOperation(
	ctx context.Context,
	backend executor.Backend,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
	operations OperationRegistry,
	optimizers OptimizerRegistry,
) (tensor.Float64Tensor, error) {
	optimizer, optimizerErr := BuildOptimizer(optimizers, node)

	if optimizerErr == nil {
		return executor.RunOptimizer(ctx, backend, node, inputs, optimizer)
	}

	operation, operationErr := BuildOperation(operations, node)

	if operationErr == nil {
		return executor.RunOperation(ctx, backend, node, inputs, operation)
	}

	if IsOptimizerOperation(node.Op) {
		return nil, optimizerErr
	}

	return nil, operationErr
}

func BuildOperation(
	registry OperationRegistry,
	node executor.NodeSpec,
) (state.Operation, error) {
	if registry == nil {
		return nil, fmt.Errorf("dispatch: operation registry is required")
	}

	config := executor.OperationConfig(node)

	switch executor.NormalizeOperation(node.Op) {
	case ir.OpAdd:
		return registry.Add(config)
	case ir.OpMul:
		return registry.Mul(config)
	case ir.OpMatmul:
		return registry.Matmul(config)
	case ir.OpReLU:
		return registry.ReLU(config)
	case ir.OpLeakyReLU:
		return registry.LeakyReLU(config)
	case ir.OpGELU:
		return registry.GELU(config)
	case ir.OpTanh:
		return registry.Tanh(config)
	case ir.OpSigmoid:
		return registry.Sigmoid(config)
	case ir.OpSwiGLU:
		return registry.SwiGLU(config)
	case ir.OpSwish:
		return registry.Swish(config)
	case ir.OpSELU:
		return registry.SELU(config)
	}

	switch strings.ToLower(string(node.Op)) {
	case "activation.swish", "swish":
		return registry.Swish(config)
	case "activation.selu", "selu":
		return registry.SELU(config)
	case "attention.sdpa":
		return registry.SDPA(config)
	case "attention.mqa":
		return registry.MQA(config)
	case "attention.gqa":
		return registry.GQA(config)
	case "attention.sliding_window":
		return registry.SlidingWindowAttention(config)
	case "masking.apply":
		return registry.ApplyMask(config)
	case "masking.causal":
		return registry.CausalMask(config)
	case "math.exp":
		return registry.Exp(config)
	case "math.sin":
		return registry.Sin(config)
	case "math.cos":
		return registry.Cos(config)
	case "math.log":
		return registry.Log(config)
	case "math.logsumexp":
		return registry.LogSumExp(config)
	case "math.softmax":
		return registry.Softmax(config)
	case "math.outer":
		return registry.Outer(config)
	case "math.sign":
		return registry.Sign(config)
	case "math.inv_sqrt_dim_scale":
		return registry.InvSqrtDimScale(config)
	case "math.dropout":
		return registry.Dropout(config)
	case "math.rmsnorm":
		return registry.RMSNorm(config)
	case "math.layernorm":
		return registry.LayerNorm(config)
	case "math.groupnorm":
		return registry.GroupNorm(config)
	case "shape.reshape":
		return registry.Reshape(config)
	case "shape.transpose":
		return registry.Transpose(config)
	case "shape.concat":
		return registry.Concat(config)
	case "shape.split":
		return registry.Split(config)
	case "shape.upsample_nearest2d":
		return registry.UpsampleNearest2D(config)
	case "shape.view_as_heads":
		return registry.ViewAsHeads(config)
	case "shape.merge_heads":
		return registry.MergeHeads(config)
	case "shape.last_token":
		return registry.LastToken(config)
	case "shape.slice":
		return registry.Slice(config)
	case "positional.rope":
		return registry.RoPE(config)
	case "positional.alibi":
		return registry.ALiBi(config)
	case "embedding.token":
		return registry.TokenEmbedding(config)
	case "convolution.conv1d":
		return registry.Conv1D(config)
	case "convolution.conv2d":
		return registry.Conv2D(config)
	case "convolution.conv3d":
		return registry.Conv3D(config)
	case "convolution.conv_transpose2d":
		return registry.ConvTranspose2D(config)
	case "pooling.max_pool2d":
		return registry.MaxPool2D(config)
	case "pooling.avg_pool2d":
		return registry.AvgPool2D(config)
	case "pooling.adaptive_avg_pool2d":
		return registry.AdaptiveAvgPool2D(config)
	case "pooling.adaptive_max_pool2d":
		return registry.AdaptiveMaxPool2D(config)
	case "projection.linear":
		return registry.Linear(config)
	case "projection.fused_qkv":
		return registry.FusedQKV(config)
	case "hawkes.intensity":
		return registry.HawkesIntensity(config)
	case "hawkes.kernel_matrix":
		return registry.HawkesKernelMatrix(config)
	case "hawkes.log_likelihood":
		return registry.HawkesLogLikelihood(config)
	case "hawkes.simulate":
		return registry.HawkesSimulate(config)
	case "vsa.bind":
		return registry.VSABind(config)
	case "vsa.bundle":
		return registry.VSABundle(config)
	case "vsa.similarity":
		return registry.VSASimilarity(config)
	case "vsa.permute":
		return registry.VSAPermute(config)
	case "vsa.inverse_permute":
		return registry.VSAInversePermute(config)
	case "active_inference.belief_update":
		return registry.BeliefUpdate(config)
	case "active_inference.expected_free_energy":
		return registry.ExpectedFreeEnergy(config)
	case "active_inference.free_energy":
		return registry.FreeEnergy(config)
	case "active_inference.precision_weight":
		return registry.PrecisionWeight(config)
	case "predictive_coding.prediction":
		return registry.Prediction(config)
	case "predictive_coding.prediction_error":
		return registry.PredictionError(config)
	case "predictive_coding.update_representation":
		return registry.UpdateRepresentation(config)
	case "predictive_coding.update_weights":
		return registry.UpdateWeights(config)
	case "markov_blanket.flow_active":
		return registry.FlowActive(config)
	case "markov_blanket.flow_internal":
		return registry.FlowInternal(config)
	case "markov_blanket.mutual_information":
		return registry.MutualInformation(config)
	case "markov_blanket.partition":
		return registry.Partition(config)
	case "causal.backdoor_adjustment":
		return registry.BackdoorAdjustment(config)
	case "causal.cate":
		return registry.CATE(config)
	case "causal.counterfactual":
		return registry.Counterfactual(config)
	case "causal.dag_markov_factorization":
		return registry.DAGMarkovFactorization(config)
	case "causal.do_calculus":
		return registry.DoCalculus(config)
	case "causal.frontdoor_adjustment":
		return registry.FrontdoorAdjustment(config)
	case "causal.iv_estimate":
		return registry.IVEstimate(config)
	case "train.loss.mse":
		return registry.MSELoss(config)
	case "train.loss.cross_entropy":
		return registry.CrossEntropyLoss(config)
	case "train.loss.mse_grad", "train.grad.mse":
		return registry.MSEGrad(config)
	case "train.loss.cross_entropy_grad", "train.grad.cross_entropy":
		return registry.CrossEntropyGrad(config)
	case "bench.accuracy", "bench.metric.accuracy":
		return registry.Accuracy(config)
	case "bench.perplexity", "bench.metric.perplexity":
		return registry.Perplexity(config)
	case "bench.f1", "bench.metric.f1":
		return registry.F1(config)
	case "model.graft":
		return registry.Graft(config)
	case "model.freeze":
		return registry.Freeze(config)
	default:
		return nil, fmt.Errorf("dispatch: unsupported operation %q", node.Op)
	}
}

func BuildOptimizer(
	registry OptimizerRegistry,
	node executor.NodeSpec,
) (state.Optimizer, error) {
	if !IsOptimizerOperation(node.Op) {
		return nil, fmt.Errorf("dispatch: operation %q is not an optimizer", node.Op)
	}

	if registry == nil {
		return nil, fmt.Errorf("dispatch: optimizer registry is required")
	}

	config := executor.OperationConfig(node)

	switch strings.ToLower(string(node.Op)) {
	case "train.optimizer.adam", "optimizer.adam":
		return registry.Adam(config)
	case "train.optimizer.adamw", "optimizer.adamw":
		return registry.AdamW(config)
	case "train.optimizer.adamax", "optimizer.adamax":
		return registry.AdaMax(config)
	case "train.optimizer.sgd", "optimizer.sgd":
		return registry.SGD(config)
	case "train.optimizer.lion", "optimizer.lion":
		return registry.Lion(config)
	case "train.optimizer.rmsprop", "optimizer.rmsprop":
		return registry.RMSProp(config)
	case "train.optimizer.hebbian", "optimizer.hebbian":
		return registry.Hebbian(config)
	case "train.optimizer.lars", "optimizer.lars":
		return registry.Lars(config)
	case "train.optimizer.lamb", "optimizer.lamb":
		return registry.Lamb(config)
	case "train.optimizer.adagrad", "optimizer.adagrad":
		return registry.AdaGrad(config)
	case "train.optimizer.adadelta", "optimizer.adadelta":
		return registry.AdaDelta(config)
	case "train.optimizer.lbfgs", "optimizer.lbfgs":
		return registry.LBFGS(config)
	default:
		return nil, fmt.Errorf("dispatch: unsupported optimizer operation %q", node.Op)
	}
}

func IsOptimizerOperation(operationID ir.OpType) bool {
	switch strings.ToLower(string(operationID)) {
	case "train.optimizer.adam",
		"optimizer.adam",
		"train.optimizer.adamw",
		"optimizer.adamw",
		"train.optimizer.adamax",
		"optimizer.adamax",
		"train.optimizer.sgd",
		"optimizer.sgd",
		"train.optimizer.lion",
		"optimizer.lion",
		"train.optimizer.rmsprop",
		"optimizer.rmsprop",
		"train.optimizer.hebbian",
		"optimizer.hebbian",
		"train.optimizer.lars",
		"optimizer.lars",
		"train.optimizer.lamb",
		"optimizer.lamb",
		"train.optimizer.adagrad",
		"optimizer.adagrad",
		"train.optimizer.adadelta",
		"optimizer.adadelta",
		"train.optimizer.lbfgs",
		"optimizer.lbfgs":
		return true
	default:
		return false
	}
}
