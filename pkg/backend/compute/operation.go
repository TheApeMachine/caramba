package compute

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type Operation = state.Operation

/*
OperationRegistry exposes backend-native operation constructors with the same
state-dict-first shape as OptimizerRegistry.
*/
type OperationRegistry interface {
	ActivationOperationRegistry
	AttentionOperationRegistry
	MaskingOperationRegistry
	MathOperationRegistry
	ShapeOperationRegistry
	PositionalOperationRegistry
	EmbeddingOperationRegistry
	ConvolutionOperationRegistry
	PoolingOperationRegistry
	ProjectionOperationRegistry
	HawkesOperationRegistry
	VSAOperationRegistry
	ActiveInferenceOperationRegistry
	PredictiveCodingOperationRegistry
	MarkovBlanketOperationRegistry
	CausalOperationRegistry
	TrainingOperationRegistry
	BenchmarkOperationRegistry
	ModelOperationRegistry
}

type ActivationOperationRegistry interface {
	ReLU(*state.Dict) (Operation, error)
	LeakyReLU(*state.Dict) (Operation, error)
	GELU(*state.Dict) (Operation, error)
	Tanh(*state.Dict) (Operation, error)
	Sigmoid(*state.Dict) (Operation, error)
	SwiGLU(*state.Dict) (Operation, error)
	Swish(*state.Dict) (Operation, error)
	SELU(*state.Dict) (Operation, error)
}

type AttentionOperationRegistry interface {
	SDPA(*state.Dict) (Operation, error)
	MQA(*state.Dict) (Operation, error)
	GQA(*state.Dict) (Operation, error)
	SlidingWindowAttention(*state.Dict) (Operation, error)
}

type MaskingOperationRegistry interface {
	ApplyMask(*state.Dict) (Operation, error)
	CausalMask(*state.Dict) (Operation, error)
}

type MathOperationRegistry interface {
	Add(*state.Dict) (Operation, error)
	Mul(*state.Dict) (Operation, error)
	Matmul(*state.Dict) (Operation, error)
	Exp(*state.Dict) (Operation, error)
	Log(*state.Dict) (Operation, error)
	LogSumExp(*state.Dict) (Operation, error)
	Softmax(*state.Dict) (Operation, error)
	Outer(*state.Dict) (Operation, error)
	Sign(*state.Dict) (Operation, error)
	InvSqrtDimScale(*state.Dict) (Operation, error)
	Dropout(*state.Dict) (Operation, error)
	RMSNorm(*state.Dict) (Operation, error)
	LayerNorm(*state.Dict) (Operation, error)
}

type ShapeOperationRegistry interface {
	Reshape(*state.Dict) (Operation, error)
	Transpose(*state.Dict) (Operation, error)
	Concat(*state.Dict) (Operation, error)
	Split(*state.Dict) (Operation, error)
	ViewAsHeads(*state.Dict) (Operation, error)
	MergeHeads(*state.Dict) (Operation, error)
	LastToken(*state.Dict) (Operation, error)
}

type PositionalOperationRegistry interface {
	RoPE(*state.Dict) (Operation, error)
	ALiBi(*state.Dict) (Operation, error)
}

type EmbeddingOperationRegistry interface {
	TokenEmbedding(*state.Dict) (Operation, error)
}

type ConvolutionOperationRegistry interface {
	Conv1D(*state.Dict) (Operation, error)
	Conv2D(*state.Dict) (Operation, error)
	Conv3D(*state.Dict) (Operation, error)
	ConvTranspose2D(*state.Dict) (Operation, error)
}

type PoolingOperationRegistry interface {
	MaxPool2D(*state.Dict) (Operation, error)
	AvgPool2D(*state.Dict) (Operation, error)
	AdaptiveAvgPool2D(*state.Dict) (Operation, error)
	AdaptiveMaxPool2D(*state.Dict) (Operation, error)
}

type ProjectionOperationRegistry interface {
	Linear(*state.Dict) (Operation, error)
	FusedQKV(*state.Dict) (Operation, error)
}

type HawkesOperationRegistry interface {
	HawkesIntensity(*state.Dict) (Operation, error)
	HawkesKernelMatrix(*state.Dict) (Operation, error)
	HawkesLogLikelihood(*state.Dict) (Operation, error)
	HawkesSimulate(*state.Dict) (Operation, error)
}

type VSAOperationRegistry interface {
	VSABind(*state.Dict) (Operation, error)
	VSABundle(*state.Dict) (Operation, error)
	VSASimilarity(*state.Dict) (Operation, error)
	VSAPermute(*state.Dict) (Operation, error)
	VSAInversePermute(*state.Dict) (Operation, error)
}

type ActiveInferenceOperationRegistry interface {
	BeliefUpdate(*state.Dict) (Operation, error)
	ExpectedFreeEnergy(*state.Dict) (Operation, error)
	FreeEnergy(*state.Dict) (Operation, error)
	PrecisionWeight(*state.Dict) (Operation, error)
}

type PredictiveCodingOperationRegistry interface {
	Prediction(*state.Dict) (Operation, error)
	PredictionError(*state.Dict) (Operation, error)
	UpdateRepresentation(*state.Dict) (Operation, error)
	UpdateWeights(*state.Dict) (Operation, error)
}

type MarkovBlanketOperationRegistry interface {
	FlowActive(*state.Dict) (Operation, error)
	FlowInternal(*state.Dict) (Operation, error)
	MutualInformation(*state.Dict) (Operation, error)
	Partition(*state.Dict) (Operation, error)
}

type CausalOperationRegistry interface {
	BackdoorAdjustment(*state.Dict) (Operation, error)
	CATE(*state.Dict) (Operation, error)
	Counterfactual(*state.Dict) (Operation, error)
	DAGMarkovFactorization(*state.Dict) (Operation, error)
	DoCalculus(*state.Dict) (Operation, error)
	FrontdoorAdjustment(*state.Dict) (Operation, error)
	IVEstimate(*state.Dict) (Operation, error)
}

type TrainingOperationRegistry interface {
	MSELoss(*state.Dict) (Operation, error)
	CrossEntropyLoss(*state.Dict) (Operation, error)
	MSEGrad(*state.Dict) (Operation, error)
	CrossEntropyGrad(*state.Dict) (Operation, error)
}

type BenchmarkOperationRegistry interface {
	Accuracy(*state.Dict) (Operation, error)
	Perplexity(*state.Dict) (Operation, error)
	F1(*state.Dict) (Operation, error)
}

type ModelOperationRegistry interface {
	Load(*state.Dict) (Operation, error)
	Surgery(*state.Dict) (Operation, error)
	Graft(*state.Dict) (Operation, error)
	LoRA(*state.Dict) (Operation, error)
	Adapter(*state.Dict) (Operation, error)
	Freeze(*state.Dict) (Operation, error)
}

func NewOperationRegistry(registryType OperationRegistry) OperationRegistry {
	return registryType
}
