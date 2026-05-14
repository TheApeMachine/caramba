//go:build !linux || !cgo || !cuda

package cuda

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

func (registry OperationRegistry) ReLU(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) LeakyReLU(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) GELU(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Tanh(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Sigmoid(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) SwiGLU(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Swish(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) SDPA(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) MQA(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) GQA(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) SlidingWindowAttention(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) ApplyMask(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) CausalMask(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Add(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Mul(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Matmul(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Exp(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Log(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) LogSumExp(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Softmax(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Outer(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Sign(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) InvSqrtDimScale(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Dropout(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) RMSNorm(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) LayerNorm(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Reshape(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Transpose(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Concat(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Split(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) ViewAsHeads(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) MergeHeads(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) RoPE(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) ALiBi(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) TokenEmbedding(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Conv1D(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Conv2D(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Conv3D(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) ConvTranspose2D(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) MaxPool2D(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) AvgPool2D(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) AdaptiveAvgPool2D(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) AdaptiveMaxPool2D(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Linear(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) FusedQKV(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) HawkesIntensity(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) HawkesKernelMatrix(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) HawkesLogLikelihood(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) HawkesSimulate(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) VSABind(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) VSABundle(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) VSASimilarity(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) VSAPermute(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) VSAInversePermute(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) BeliefUpdate(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) ExpectedFreeEnergy(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) FreeEnergy(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) PrecisionWeight(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Prediction(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) PredictionError(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) UpdateRepresentation(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) UpdateWeights(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) FlowActive(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) FlowInternal(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) MutualInformation(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Partition(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) BackdoorAdjustment(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) CATE(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Counterfactual(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) DAGMarkovFactorization(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) DoCalculus(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) FrontdoorAdjustment(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) IVEstimate(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) MSELoss(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) CrossEntropyLoss(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) MSEGrad(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) CrossEntropyGrad(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Accuracy(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Perplexity(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) F1(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Load(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Surgery(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Graft(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) LoRA(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Adapter(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry OperationRegistry) Freeze(*state.Dict) (state.Operation, error) {
	return nil, cudaOptimizerUnavailable()
}
