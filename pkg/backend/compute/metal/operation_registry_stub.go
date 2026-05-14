//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

func unavailableOperation() (state.Operation, error) { return nil, metalUnavailable() }

func (registry *OperationRegistry) ReLU(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) LeakyReLU(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) GELU(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Tanh(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Sigmoid(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) SwiGLU(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Swish(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) SDPA(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) MQA(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) GQA(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) SlidingWindowAttention(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) ApplyMask(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) CausalMask(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Add(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Mul(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Matmul(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Exp(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Log(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) LogSumExp(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Softmax(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Outer(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Sign(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) InvSqrtDimScale(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Dropout(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) RMSNorm(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) LayerNorm(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Reshape(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Transpose(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Concat(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Split(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) ViewAsHeads(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) MergeHeads(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) RoPE(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) ALiBi(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) TokenEmbedding(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Conv1D(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Conv2D(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Conv3D(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) ConvTranspose2D(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) MaxPool2D(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) AvgPool2D(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) AdaptiveAvgPool2D(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) AdaptiveMaxPool2D(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Linear(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) FusedQKV(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) HawkesIntensity(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) HawkesKernelMatrix(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) HawkesLogLikelihood(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) HawkesSimulate(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) VSABind(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) VSABundle(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) VSASimilarity(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) VSAPermute(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) VSAInversePermute(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) BeliefUpdate(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) ExpectedFreeEnergy(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) FreeEnergy(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) PrecisionWeight(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Prediction(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) PredictionError(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) UpdateRepresentation(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) UpdateWeights(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) FlowActive(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) FlowInternal(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) MutualInformation(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Partition(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) BackdoorAdjustment(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) CATE(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Counterfactual(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) DAGMarkovFactorization(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) DoCalculus(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) FrontdoorAdjustment(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) IVEstimate(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) MSELoss(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) CrossEntropyLoss(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) MSEGrad(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) CrossEntropyGrad(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Accuracy(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Perplexity(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) F1(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Load(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Surgery(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Graft(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) LoRA(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Adapter(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}

func (registry *OperationRegistry) Freeze(_ *state.Dict) (state.Operation, error) {
	return unavailableOperation()
}
