package operation

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/activation"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/active_inference"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/attention"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/bench"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/convolution"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/embedding"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/hawkes"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/markov_blanket"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/masking"
	mathop "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/pooling"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/positional"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/predictive_coding"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/projection"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/shape"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/train"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/vsa"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type OperationRegistry struct{}

func NewOperationRegistry() *OperationRegistry {
	return &OperationRegistry{}
}

func (registry *OperationRegistry) ReLU(*state.Dict) (state.Operation, error) {
	return activation.NewReLU(), nil
}

func (registry *OperationRegistry) LeakyReLU(*state.Dict) (state.Operation, error) {
	return activation.NewLeakyReLU(), nil
}

func (registry *OperationRegistry) GELU(*state.Dict) (state.Operation, error) {
	return activation.NewGelu(), nil
}

func (registry *OperationRegistry) Tanh(*state.Dict) (state.Operation, error) {
	return activation.NewTanh(), nil
}

func (registry *OperationRegistry) Sigmoid(*state.Dict) (state.Operation, error) {
	return activation.NewSigmoid(), nil
}

func (registry *OperationRegistry) SwiGLU(*state.Dict) (state.Operation, error) {
	return activation.NewSwiGLU(), nil
}

func (registry *OperationRegistry) Swish(*state.Dict) (state.Operation, error) {
	return activation.NewSwish(), nil
}

func (registry *OperationRegistry) SDPA(*state.Dict) (state.Operation, error) {
	return attention.NewSDPA(), nil
}

func (registry *OperationRegistry) MQA(*state.Dict) (state.Operation, error) {
	return attention.NewMQA(), nil
}

func (registry *OperationRegistry) GQA(*state.Dict) (state.Operation, error) {
	return attention.NewGQA(), nil
}

func (registry *OperationRegistry) SlidingWindowAttention(*state.Dict) (state.Operation, error) {
	return attention.NewSlidingWindow(), nil
}

func (registry *OperationRegistry) ApplyMask(*state.Dict) (state.Operation, error) {
	return masking.NewApplyMask(), nil
}

func (registry *OperationRegistry) CausalMask(*state.Dict) (state.Operation, error) {
	return masking.NewCausalMask(), nil
}

func (registry *OperationRegistry) Add(*state.Dict) (state.Operation, error) {
	return mathop.NewAdd(), nil
}

func (registry *OperationRegistry) Mul(*state.Dict) (state.Operation, error) {
	return mathop.NewMul(), nil
}

func (registry *OperationRegistry) Matmul(*state.Dict) (state.Operation, error) {
	return mathop.NewMatmul(), nil
}

func (registry *OperationRegistry) Exp(*state.Dict) (state.Operation, error) {
	return mathop.NewExp(), nil
}

func (registry *OperationRegistry) Log(*state.Dict) (state.Operation, error) {
	return mathop.NewLog(), nil
}

func (registry *OperationRegistry) LogSumExp(*state.Dict) (state.Operation, error) {
	return mathop.NewLogSumExp(), nil
}

func (registry *OperationRegistry) Softmax(*state.Dict) (state.Operation, error) {
	return mathop.NewSoftmax(), nil
}

func (registry *OperationRegistry) Outer(*state.Dict) (state.Operation, error) {
	return mathop.NewOuter(), nil
}

func (registry *OperationRegistry) Sign(*state.Dict) (state.Operation, error) {
	return mathop.NewSign(), nil
}

func (registry *OperationRegistry) InvSqrtDimScale(*state.Dict) (state.Operation, error) {
	return mathop.NewInvSqrtDimScale(), nil
}

func (registry *OperationRegistry) Dropout(*state.Dict) (state.Operation, error) {
	return mathop.NewDropout(), nil
}

func (registry *OperationRegistry) RMSNorm(*state.Dict) (state.Operation, error) {
	return mathop.NewRMSNorm(), nil
}

func (registry *OperationRegistry) LayerNorm(*state.Dict) (state.Operation, error) {
	return mathop.NewLayerNorm(), nil
}

func (registry *OperationRegistry) Reshape(*state.Dict) (state.Operation, error) {
	return shape.NewReshape(), nil
}

func (registry *OperationRegistry) Transpose(*state.Dict) (state.Operation, error) {
	return shape.NewTranspose(), nil
}

func (registry *OperationRegistry) Concat(*state.Dict) (state.Operation, error) {
	return shape.NewConcat(), nil
}

func (registry *OperationRegistry) Split(*state.Dict) (state.Operation, error) {
	return shape.NewSplit(), nil
}

func (registry *OperationRegistry) ViewAsHeads(*state.Dict) (state.Operation, error) {
	return shape.NewViewAsHeads(), nil
}

func (registry *OperationRegistry) MergeHeads(*state.Dict) (state.Operation, error) {
	return shape.NewMergeHeads(), nil
}

func (registry *OperationRegistry) RoPE(*state.Dict) (state.Operation, error) {
	return positional.NewRoPE(), nil
}

func (registry *OperationRegistry) ALiBi(*state.Dict) (state.Operation, error) {
	return positional.NewALiBi(), nil
}

func (registry *OperationRegistry) TokenEmbedding(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return embedding.NewTokenEmbedding(config.VocabSize, config.DModel, config.InitStd), nil
}

func (registry *OperationRegistry) Conv1D(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return convolution.NewConv1d(
		defaultInt(config.InChannels, 1),
		defaultInt(config.OutChannels, 1),
		defaultInt(config.KernelSize, 1),
		defaultInt(config.Stride, 1),
		config.Padding,
		defaultInt(config.Dilation, 1),
		defaultInt(config.Groups, 1),
	), nil
}

func (registry *OperationRegistry) Conv2D(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return convolution.NewConv2d(
		defaultInt(config.InChannels, 1),
		defaultInt(config.OutChannels, 1),
		defaultInt(config.KernelH, 1),
		defaultInt(config.KernelW, 1),
		defaultInt(config.StrideH, 1),
		defaultInt(config.StrideW, 1),
		config.PadH,
		config.PadW,
		defaultInt(config.DilationH, 1),
		defaultInt(config.DilationW, 1),
		defaultInt(config.Groups, 1),
	), nil
}

func (registry *OperationRegistry) Conv3D(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return convolution.NewConv3d(
		defaultInt(config.InChannels, 1),
		defaultInt(config.OutChannels, 1),
		defaultInt(config.KernelD, 1),
		defaultInt(config.KernelH, 1),
		defaultInt(config.KernelW, 1),
		defaultInt(config.StrideD, 1),
		defaultInt(config.StrideH, 1),
		defaultInt(config.StrideW, 1),
		config.PadD,
		config.PadH,
		config.PadW,
		defaultInt(config.DilationD, 1),
		defaultInt(config.DilationH, 1),
		defaultInt(config.DilationW, 1),
		defaultInt(config.Groups, 1),
	), nil
}

func (registry *OperationRegistry) ConvTranspose2D(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return convolution.NewConvTranspose2d(
		defaultInt(config.InChannels, 1),
		defaultInt(config.OutChannels, 1),
		defaultInt(config.KernelH, 1),
		defaultInt(config.KernelW, 1),
		defaultInt(config.StrideH, 1),
		defaultInt(config.StrideW, 1),
		config.PadH,
		config.PadW,
		config.OutPadH,
		config.OutPadW,
		defaultInt(config.DilationH, 1),
		defaultInt(config.DilationW, 1),
		defaultInt(config.Groups, 1),
	), nil
}

func (registry *OperationRegistry) MaxPool2D(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return pooling.NewMaxPool2d(
		defaultInt(config.KernelH, 1),
		defaultInt(config.KernelW, 1),
		defaultInt(config.StrideH, 1),
		defaultInt(config.StrideW, 1),
		config.PadH,
		config.PadW,
		defaultInt(config.DilationH, 1),
		defaultInt(config.DilationW, 1),
		config.Ceil,
	), nil
}

func (registry *OperationRegistry) AvgPool2D(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return pooling.NewAvgPool2d(
		defaultInt(config.KernelH, 1),
		defaultInt(config.KernelW, 1),
		defaultInt(config.StrideH, 1),
		defaultInt(config.StrideW, 1),
		config.PadH,
		config.PadW,
		defaultInt(config.DilationH, 1),
		defaultInt(config.DilationW, 1),
		config.Ceil,
		config.CountPad,
		config.Divisor,
	), nil
}

func (registry *OperationRegistry) AdaptiveAvgPool2D(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return pooling.NewAdaptiveAvgPool2d(
		defaultInt(config.OutH, 1),
		defaultInt(config.OutW, 1),
	), nil
}

func (registry *OperationRegistry) AdaptiveMaxPool2D(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return pooling.NewAdaptiveMaxPool2d(
		defaultInt(config.OutH, 1),
		defaultInt(config.OutW, 1),
	), nil
}

func (registry *OperationRegistry) Linear(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return projection.NewLinear(config.InFeatures, config.OutFeatures), nil
}

func (registry *OperationRegistry) FusedQKV(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return projection.NewFusedQKV(config.DIn, config.DQ, config.DK, config.DV), nil
}

func (registry *OperationRegistry) HawkesIntensity(*state.Dict) (state.Operation, error) {
	return hawkes.NewIntensity(), nil
}

func (registry *OperationRegistry) HawkesKernelMatrix(*state.Dict) (state.Operation, error) {
	return hawkes.NewKernelMatrix(), nil
}

func (registry *OperationRegistry) HawkesLogLikelihood(*state.Dict) (state.Operation, error) {
	return hawkes.NewLogLikelihood(), nil
}

func (registry *OperationRegistry) HawkesSimulate(*state.Dict) (state.Operation, error) {
	return hawkes.NewSimulate(), nil
}

func (registry *OperationRegistry) VSABind(*state.Dict) (state.Operation, error) {
	return vsa.NewBind(), nil
}

func (registry *OperationRegistry) VSABundle(*state.Dict) (state.Operation, error) {
	return vsa.NewBundle(), nil
}

func (registry *OperationRegistry) VSASimilarity(*state.Dict) (state.Operation, error) {
	return vsa.NewSimilarity(), nil
}

func (registry *OperationRegistry) VSAPermute(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return vsa.NewPermute(config.K), nil
}

func (registry *OperationRegistry) VSAInversePermute(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return vsa.NewInversePermute(config.K), nil
}

func (registry *OperationRegistry) BeliefUpdate(*state.Dict) (state.Operation, error) {
	return active_inference.NewBeliefUpdate(), nil
}

func (registry *OperationRegistry) ExpectedFreeEnergy(*state.Dict) (state.Operation, error) {
	return active_inference.NewExpectedFreeEnergy(), nil
}

func (registry *OperationRegistry) FreeEnergy(*state.Dict) (state.Operation, error) {
	return active_inference.NewFreeEnergy(), nil
}

func (registry *OperationRegistry) PrecisionWeight(*state.Dict) (state.Operation, error) {
	return active_inference.NewPrecisionWeight(), nil
}

func (registry *OperationRegistry) Prediction(*state.Dict) (state.Operation, error) {
	return predictive_coding.NewPrediction(), nil
}

func (registry *OperationRegistry) PredictionError(*state.Dict) (state.Operation, error) {
	return predictive_coding.NewPredictionError(), nil
}

func (registry *OperationRegistry) UpdateRepresentation(*state.Dict) (state.Operation, error) {
	return predictive_coding.NewUpdateRepresentation(), nil
}

func (registry *OperationRegistry) UpdateWeights(*state.Dict) (state.Operation, error) {
	return predictive_coding.NewUpdateWeights(), nil
}

func (registry *OperationRegistry) FlowActive(*state.Dict) (state.Operation, error) {
	return markov_blanket.NewFlowActive(), nil
}

func (registry *OperationRegistry) FlowInternal(*state.Dict) (state.Operation, error) {
	return markov_blanket.NewFlowInternal(), nil
}

func (registry *OperationRegistry) MutualInformation(*state.Dict) (state.Operation, error) {
	return markov_blanket.NewMutualInformation(), nil
}

func (registry *OperationRegistry) Partition(*state.Dict) (state.Operation, error) {
	return markov_blanket.NewPartition(), nil
}

func (registry *OperationRegistry) BackdoorAdjustment(*state.Dict) (state.Operation, error) {
	return causal.NewBackdoorAdjustment(), nil
}

func (registry *OperationRegistry) CATE(*state.Dict) (state.Operation, error) {
	return causal.NewCATE(), nil
}

func (registry *OperationRegistry) Counterfactual(*state.Dict) (state.Operation, error) {
	return causal.NewCounterfactual(), nil
}

func (registry *OperationRegistry) DAGMarkovFactorization(*state.Dict) (state.Operation, error) {
	return causal.NewDAGMarkovFactorization(), nil
}

func (registry *OperationRegistry) DoCalculus(*state.Dict) (state.Operation, error) {
	return causal.NewDoCalculus(), nil
}

func (registry *OperationRegistry) FrontdoorAdjustment(*state.Dict) (state.Operation, error) {
	return causal.NewFrontdoorAdjustment(), nil
}

func (registry *OperationRegistry) IVEstimate(*state.Dict) (state.Operation, error) {
	return causal.NewIVEstimate(), nil
}

func (registry *OperationRegistry) MSELoss(*state.Dict) (state.Operation, error) {
	return train.NewMSELoss(), nil
}

func (registry *OperationRegistry) CrossEntropyLoss(*state.Dict) (state.Operation, error) {
	return train.NewCrossEntropyLoss(), nil
}

func (registry *OperationRegistry) MSEGrad(*state.Dict) (state.Operation, error) {
	return train.NewMSEGrad(), nil
}

func (registry *OperationRegistry) CrossEntropyGrad(*state.Dict) (state.Operation, error) {
	return train.NewCrossEntropyGrad(), nil
}

func (registry *OperationRegistry) Accuracy(*state.Dict) (state.Operation, error) {
	return bench.NewAccuracy(), nil
}

func (registry *OperationRegistry) Perplexity(*state.Dict) (state.Operation, error) {
	return bench.NewPerplexity(), nil
}

func (registry *OperationRegistry) F1(*state.Dict) (state.Operation, error) {
	return bench.NewF1(), nil
}

func (registry *OperationRegistry) Load(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return model.NewLoader(config.Source, config.File, config.Cache), nil
}

func (registry *OperationRegistry) Surgery(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return model.NewSurgery(
		config.Source, config.Op, config.At, config.After, config.Name, config.Layer,
	), nil
}

func (registry *OperationRegistry) Graft(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return model.NewGraft(config.Source, config.At, config.Mode), nil
}

func (registry *OperationRegistry) LoRA(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return model.NewLoRA(config.Source, config.Preset, config.Targets, config.Rank, config.Alpha), nil
}

func (registry *OperationRegistry) Adapter(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return model.NewAdapter(config.Source, config.At, config.Reduction), nil
}

func (registry *OperationRegistry) Freeze(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return model.NewFreeze(config.Source, config.Pattern, config.Except, config.Frozen), nil
}

func stateConfig(config *state.Dict) *state.Dict {
	if config == nil {
		return state.NewDict()
	}

	return config
}

func defaultInt(value, fallback int) int {
	if value == 0 {
		return fallback
	}

	return value
}
