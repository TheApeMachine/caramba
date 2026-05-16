//go:build darwin && cgo

package metal

import (
	"fmt"
	stdmath "math"
	"path/filepath"
	"runtime"

	"github.com/theapemachine/caramba/pkg/backend/compute/rotary"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func (registry OperationRegistry) ReLU(config *state.Dict) (state.Operation, error) {
	activation, err := newMetalActivation(config)

	if err != nil {
		return nil, err
	}

	return &ReLU{activation: activation}, nil
}

func (registry OperationRegistry) LeakyReLU(config *state.Dict) (state.Operation, error) {
	activation, err := newMetalActivation(config)

	if err != nil {
		return nil, err
	}

	return &LeakyReLU{activation: activation}, nil
}

func (registry OperationRegistry) GELU(config *state.Dict) (state.Operation, error) {
	activation, err := newMetalActivation(config)

	if err != nil {
		return nil, err
	}

	return &GELU{activation: activation}, nil
}

func (registry OperationRegistry) Tanh(config *state.Dict) (state.Operation, error) {
	activation, err := newMetalActivation(config)

	if err != nil {
		return nil, err
	}

	return &Tanh{activation: activation}, nil
}

func (registry OperationRegistry) Sigmoid(config *state.Dict) (state.Operation, error) {
	activation, err := newMetalActivation(config)

	if err != nil {
		return nil, err
	}

	return &Sigmoid{activation: activation}, nil
}

func (registry OperationRegistry) SwiGLU(config *state.Dict) (state.Operation, error) {
	activation, err := newMetalActivation(config)

	if err != nil {
		return nil, err
	}

	return &SwiGLU{activation: activation}, nil
}

func (registry OperationRegistry) Swish(config *state.Dict) (state.Operation, error) {
	activation, err := newMetalActivation(config)

	if err != nil {
		return nil, err
	}

	return &Swish{activation: activation}, nil
}

func (registry OperationRegistry) SELU(config *state.Dict) (state.Operation, error) {
	activation, err := newMetalActivation(config)

	if err != nil {
		return nil, err
	}

	return &SELU{activation: activation}, nil
}

func (registry OperationRegistry) SDPA(config *state.Dict) (state.Operation, error) {
	attention, err := newMetalAttention(config)

	if err != nil {
		return nil, err
	}

	return &SDPA{attention: attention}, nil
}

func (registry OperationRegistry) MQA(config *state.Dict) (state.Operation, error) {
	attention, err := newMetalAttention(config)

	if err != nil {
		return nil, err
	}

	return &MQA{attention: attention}, nil
}

func (registry OperationRegistry) GQA(config *state.Dict) (state.Operation, error) {
	attention, err := newMetalAttention(config)

	if err != nil {
		return nil, err
	}

	return &GQA{attention: attention}, nil
}

func (registry OperationRegistry) SlidingWindowAttention(config *state.Dict) (state.Operation, error) {
	attention, err := newMetalAttention(config)

	if err != nil {
		return nil, err
	}

	return &SlidingWindowAttention{attention: attention}, nil
}

func (registry OperationRegistry) ApplyMask(config *state.Dict) (state.Operation, error) {
	masking, err := newMetalMasking(config)

	if err != nil {
		return nil, err
	}

	return &ApplyMask{masking: masking}, nil
}

func (registry OperationRegistry) CausalMask(config *state.Dict) (state.Operation, error) {
	masking, err := newMetalMasking(config)

	if err != nil {
		return nil, err
	}

	return &CausalMask{masking: masking}, nil
}

func (registry OperationRegistry) Add(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Add{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Mul(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Mul{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Matmul(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Matmul{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Exp(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Exp{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Log(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Log{mathOps: mathOps}, nil
}

func (registry OperationRegistry) LogSumExp(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &LogSumExp{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Softmax(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Softmax{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Outer(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Outer{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Sign(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Sign{mathOps: mathOps}, nil
}

func (registry OperationRegistry) InvSqrtDimScale(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &InvSqrtDimScale{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Dropout(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &Dropout{mathOps: mathOps}, nil
}

func (registry OperationRegistry) RMSNorm(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &RMSNorm{mathOps: mathOps}, nil
}

func (registry OperationRegistry) LayerNorm(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &LayerNorm{mathOps: mathOps}, nil
}

func (registry OperationRegistry) GroupNorm(config *state.Dict) (state.Operation, error) {
	mathOps, err := newMetalMath(config)

	if err != nil {
		return nil, err
	}

	return &GroupNorm{mathOps: mathOps}, nil
}

func (registry OperationRegistry) Reshape(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newMetalShape(config)

	if err != nil {
		return nil, err
	}

	return &Reshape{shapeOps: shapeOps}, nil
}

func (registry OperationRegistry) Transpose(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newMetalShape(config)

	if err != nil {
		return nil, err
	}

	return &Transpose{shapeOps: shapeOps}, nil
}

func (registry OperationRegistry) Concat(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newMetalShape(config)

	if err != nil {
		return nil, err
	}

	return &Concat{shapeOps: shapeOps}, nil
}

func (registry OperationRegistry) Split(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newMetalShape(config)

	if err != nil {
		return nil, err
	}

	return &Split{shapeOps: shapeOps}, nil
}

func (registry OperationRegistry) UpsampleNearest2D(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newMetalShape(config)

	if err != nil {
		return nil, err
	}

	return &UpsampleNearest2D{shapeOps: shapeOps}, nil
}

func (registry OperationRegistry) ViewAsHeads(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newMetalShape(config)

	if err != nil {
		return nil, err
	}

	return &ViewAsHeads{shapeOps: shapeOps}, nil
}

func (registry OperationRegistry) MergeHeads(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newMetalShape(config)

	if err != nil {
		return nil, err
	}

	return &MergeHeads{shapeOps: shapeOps}, nil
}

func (registry OperationRegistry) LastToken(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newMetalShape(config)

	if err != nil {
		return nil, err
	}

	return &LastToken{shapeOps: shapeOps}, nil
}

func (registry OperationRegistry) RoPE(config *state.Dict) (state.Operation, error) {
	positional, err := newMetalPositional(config)

	if err != nil {
		return nil, err
	}

	return &RoPE{positional: positional}, nil
}

func (registry OperationRegistry) ALiBi(config *state.Dict) (state.Operation, error) {
	positional, err := newMetalPositional(config)

	if err != nil {
		return nil, err
	}

	return &ALiBi{positional: positional}, nil
}

func (registry OperationRegistry) TokenEmbedding(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	embeddingOps, err := NewEmbeddingOps(
		metalLibrary(config, "embedding.metallib"),
		config.VocabSize,
		config.DModel,
	)

	if err != nil {
		return nil, err
	}

	return &TokenEmbedding{embeddingOps: embeddingOps}, nil
}

func (registry OperationRegistry) Conv1D(config *state.Dict) (state.Operation, error) {
	convolution, err := newMetalConvolution(config)

	if err != nil {
		return nil, err
	}

	return &Conv1D{convolution: convolution}, nil
}

func (registry OperationRegistry) Conv2D(config *state.Dict) (state.Operation, error) {
	convolution, err := newMetalConvolution(config)

	if err != nil {
		return nil, err
	}

	return &Conv2D{convolution: convolution}, nil
}

func (registry OperationRegistry) Conv3D(config *state.Dict) (state.Operation, error) {
	convolution, err := newMetalConvolution(config)

	if err != nil {
		return nil, err
	}

	return &Conv3D{convolution: convolution}, nil
}

func (registry OperationRegistry) ConvTranspose2D(config *state.Dict) (state.Operation, error) {
	convolution, err := newMetalConvolution(config)

	if err != nil {
		return nil, err
	}

	return &ConvTranspose2D{convolution: convolution}, nil
}

func (registry OperationRegistry) MaxPool2D(config *state.Dict) (state.Operation, error) {
	pooling, err := newMetalPooling(config)

	if err != nil {
		return nil, err
	}

	return &MaxPool2D{pooling: pooling}, nil
}

func (registry OperationRegistry) AvgPool2D(config *state.Dict) (state.Operation, error) {
	pooling, err := newMetalPooling(config)

	if err != nil {
		return nil, err
	}

	return &AvgPool2D{pooling: pooling}, nil
}

func (registry OperationRegistry) AdaptiveAvgPool2D(config *state.Dict) (state.Operation, error) {
	pooling, err := newMetalPooling(config)

	if err != nil {
		return nil, err
	}

	return &AdaptiveAvgPool2D{pooling: pooling}, nil
}

func (registry OperationRegistry) AdaptiveMaxPool2D(config *state.Dict) (state.Operation, error) {
	pooling, err := newMetalPooling(config)

	if err != nil {
		return nil, err
	}

	return &AdaptiveMaxPool2D{pooling: pooling}, nil
}

func (registry OperationRegistry) Linear(config *state.Dict) (state.Operation, error) {
	projection, err := newMetalProjection(config)

	if err != nil {
		return nil, err
	}

	return &Linear{projection: projection}, nil
}

func (registry OperationRegistry) FusedQKV(config *state.Dict) (state.Operation, error) {
	projection, err := newMetalProjection(config)

	if err != nil {
		return nil, err
	}

	return &FusedQKV{projection: projection}, nil
}

func (registry OperationRegistry) HawkesIntensity(config *state.Dict) (state.Operation, error) {
	hawkes, err := newMetalHawkes(config)

	if err != nil {
		return nil, err
	}

	return &HawkesIntensity{hawkes: hawkes}, nil
}

func (registry OperationRegistry) HawkesKernelMatrix(config *state.Dict) (state.Operation, error) {
	hawkes, err := newMetalHawkes(config)

	if err != nil {
		return nil, err
	}

	return &HawkesKernelMatrix{hawkes: hawkes}, nil
}

func (registry OperationRegistry) HawkesLogLikelihood(config *state.Dict) (state.Operation, error) {
	hawkes, err := newMetalHawkes(config)

	if err != nil {
		return nil, err
	}

	return &HawkesLogLikelihood{hawkes: hawkes}, nil
}

func (registry OperationRegistry) HawkesSimulate(config *state.Dict) (state.Operation, error) {
	hawkes, err := newMetalHawkes(config)

	if err != nil {
		return nil, err
	}

	return &HawkesSimulate{hawkes: hawkes}, nil
}

func (registry OperationRegistry) VSABind(config *state.Dict) (state.Operation, error) {
	vsaOps, err := newMetalVSA(config)

	if err != nil {
		return nil, err
	}

	return &VSABind{vsaOps: vsaOps}, nil
}

func (registry OperationRegistry) VSABundle(config *state.Dict) (state.Operation, error) {
	vsaOps, err := newMetalVSA(config)

	if err != nil {
		return nil, err
	}

	return &VSABundle{vsaOps: vsaOps}, nil
}

func (registry OperationRegistry) VSASimilarity(config *state.Dict) (state.Operation, error) {
	vsaOps, err := newMetalVSA(config)

	if err != nil {
		return nil, err
	}

	return &VSASimilarity{vsaOps: vsaOps}, nil
}

func (registry OperationRegistry) VSAPermute(config *state.Dict) (state.Operation, error) {
	vsaOps, err := newMetalVSA(config)

	if err != nil {
		return nil, err
	}

	return &VSAPermute{vsaOps: vsaOps}, nil
}

func (registry OperationRegistry) VSAInversePermute(config *state.Dict) (state.Operation, error) {
	vsaOps, err := newMetalVSA(config)

	if err != nil {
		return nil, err
	}

	return &VSAInversePermute{vsaOps: vsaOps}, nil
}

func (registry OperationRegistry) BeliefUpdate(config *state.Dict) (state.Operation, error) {
	activeInference, err := newMetalActiveInference(config)

	if err != nil {
		return nil, err
	}

	return &BeliefUpdate{activeInference: activeInference}, nil
}

func (registry OperationRegistry) ExpectedFreeEnergy(config *state.Dict) (state.Operation, error) {
	activeInference, err := newMetalActiveInference(config)

	if err != nil {
		return nil, err
	}

	return &ExpectedFreeEnergy{activeInference: activeInference}, nil
}

func (registry OperationRegistry) FreeEnergy(config *state.Dict) (state.Operation, error) {
	activeInference, err := newMetalActiveInference(config)

	if err != nil {
		return nil, err
	}

	return &FreeEnergy{activeInference: activeInference}, nil
}

func (registry OperationRegistry) PrecisionWeight(config *state.Dict) (state.Operation, error) {
	activeInference, err := newMetalActiveInference(config)

	if err != nil {
		return nil, err
	}

	return &PrecisionWeight{activeInference: activeInference}, nil
}

func (registry OperationRegistry) Prediction(config *state.Dict) (state.Operation, error) {
	predictiveCoding, err := newMetalPredictiveCoding(config)

	if err != nil {
		return nil, err
	}

	return &Prediction{predictiveCoding: predictiveCoding}, nil
}

func (registry OperationRegistry) PredictionError(config *state.Dict) (state.Operation, error) {
	predictiveCoding, err := newMetalPredictiveCoding(config)

	if err != nil {
		return nil, err
	}

	return &PredictionError{predictiveCoding: predictiveCoding}, nil
}

func (registry OperationRegistry) UpdateRepresentation(config *state.Dict) (state.Operation, error) {
	predictiveCoding, err := newMetalPredictiveCoding(config)

	if err != nil {
		return nil, err
	}

	return &UpdateRepresentation{predictiveCoding: predictiveCoding}, nil
}

func (registry OperationRegistry) UpdateWeights(config *state.Dict) (state.Operation, error) {
	predictiveCoding, err := newMetalPredictiveCoding(config)

	if err != nil {
		return nil, err
	}

	return &UpdateWeights{predictiveCoding: predictiveCoding}, nil
}

func (registry OperationRegistry) FlowActive(config *state.Dict) (state.Operation, error) {
	markovBlanket, err := newMetalMarkovBlanket(config)

	if err != nil {
		return nil, err
	}

	return &FlowActive{markovBlanket: markovBlanket}, nil
}

func (registry OperationRegistry) FlowInternal(config *state.Dict) (state.Operation, error) {
	markovBlanket, err := newMetalMarkovBlanket(config)

	if err != nil {
		return nil, err
	}

	return &FlowInternal{markovBlanket: markovBlanket}, nil
}

func (registry OperationRegistry) MutualInformation(config *state.Dict) (state.Operation, error) {
	markovBlanket, err := newMetalMarkovBlanket(config)

	if err != nil {
		return nil, err
	}

	return &MutualInformation{markovBlanket: markovBlanket}, nil
}

func (registry OperationRegistry) Partition(config *state.Dict) (state.Operation, error) {
	markovBlanket, err := newMetalMarkovBlanket(config)

	if err != nil {
		return nil, err
	}

	return &Partition{markovBlanket: markovBlanket}, nil
}

func (registry OperationRegistry) BackdoorAdjustment(config *state.Dict) (state.Operation, error) {
	causal, err := newMetalCausal(config)

	if err != nil {
		return nil, err
	}

	return &BackdoorAdjustment{causal: causal}, nil
}

func (registry OperationRegistry) CATE(config *state.Dict) (state.Operation, error) {
	causal, err := newMetalCausal(config)

	if err != nil {
		return nil, err
	}

	return &CATE{causal: causal}, nil
}

func (registry OperationRegistry) Counterfactual(config *state.Dict) (state.Operation, error) {
	causal, err := newMetalCausal(config)

	if err != nil {
		return nil, err
	}

	return &Counterfactual{causal: causal}, nil
}

func (registry OperationRegistry) DAGMarkovFactorization(config *state.Dict) (state.Operation, error) {
	causal, err := newMetalCausal(config)

	if err != nil {
		return nil, err
	}

	return &DAGMarkovFactorization{causal: causal}, nil
}

func (registry OperationRegistry) DoCalculus(config *state.Dict) (state.Operation, error) {
	causal, err := newMetalCausal(config)

	if err != nil {
		return nil, err
	}

	return &DoCalculus{causal: causal}, nil
}

func (registry OperationRegistry) FrontdoorAdjustment(config *state.Dict) (state.Operation, error) {
	causal, err := newMetalCausal(config)

	if err != nil {
		return nil, err
	}

	return &FrontdoorAdjustment{causal: causal}, nil
}

func (registry OperationRegistry) IVEstimate(config *state.Dict) (state.Operation, error) {
	causal, err := newMetalCausal(config)

	if err != nil {
		return nil, err
	}

	return &IVEstimate{causal: causal}, nil
}

func (registry OperationRegistry) MSELoss(config *state.Dict) (state.Operation, error) {
	trainingOps, err := newMetalTraining(config)

	if err != nil {
		return nil, err
	}

	return &MSELoss{trainingOps: trainingOps}, nil
}

func (registry OperationRegistry) CrossEntropyLoss(config *state.Dict) (state.Operation, error) {
	trainingOps, err := newMetalTraining(config)

	if err != nil {
		return nil, err
	}

	return &CrossEntropyLoss{trainingOps: trainingOps}, nil
}

func (registry OperationRegistry) MSEGrad(config *state.Dict) (state.Operation, error) {
	trainingOps, err := newMetalTraining(config)

	if err != nil {
		return nil, err
	}

	return &MSEGrad{trainingOps: trainingOps}, nil
}

func (registry OperationRegistry) CrossEntropyGrad(config *state.Dict) (state.Operation, error) {
	trainingOps, err := newMetalTraining(config)

	if err != nil {
		return nil, err
	}

	return &CrossEntropyGrad{trainingOps: trainingOps}, nil
}

func (registry OperationRegistry) Accuracy(config *state.Dict) (state.Operation, error) {
	trainingOps, err := newMetalTraining(config)

	if err != nil {
		return nil, err
	}

	return &Accuracy{trainingOps: trainingOps}, nil
}

func (registry OperationRegistry) Perplexity(config *state.Dict) (state.Operation, error) {
	trainingOps, err := newMetalTraining(config)

	if err != nil {
		return nil, err
	}

	return &Perplexity{trainingOps: trainingOps}, nil
}

func (registry OperationRegistry) F1(config *state.Dict) (state.Operation, error) {
	trainingOps, err := newMetalTraining(config)

	if err != nil {
		return nil, err
	}

	return &F1{trainingOps: trainingOps}, nil
}

func (registry OperationRegistry) Load(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return NewModelOps().NewLoader(config.Source, config.File, config.Cache), nil
}

func (registry OperationRegistry) Surgery(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return NewModelOps().NewSurgery(
		config.Source, config.Op, config.At, config.After, config.Name, config.Layer,
	), nil
}

func (registry OperationRegistry) Graft(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return NewModelOps().NewGraft(config.Source, config.At, config.Mode), nil
}

func (registry OperationRegistry) LoRA(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)

	if _, err := newMetalMath(config); err != nil {
		return nil, err
	}

	return NewModelOps().NewLoRA(
		config.Source, config.Preset, config.Targets, config.Rank, config.Alpha,
	), nil
}

func (registry OperationRegistry) Adapter(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)

	if _, err := newMetalMath(config); err != nil {
		return nil, err
	}

	return NewModelOps().NewAdapter(config.Source, config.At, config.Reduction), nil
}

func (registry OperationRegistry) Freeze(config *state.Dict) (state.Operation, error) {
	config = stateConfig(config)
	return NewModelOps().NewFreeze(config.Source, config.Pattern, config.Except, config.Frozen), nil
}

type ReLU struct{ activation *MetalActivation }
type LeakyReLU struct{ activation *MetalActivation }
type GELU struct{ activation *MetalActivation }
type Tanh struct{ activation *MetalActivation }
type Sigmoid struct{ activation *MetalActivation }
type SwiGLU struct{ activation *MetalActivation }
type Swish struct{ activation *MetalActivation }
type SELU struct{ activation *MetalActivation }

func (relu *ReLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryMetalForward(stateDict, "metal.activation.relu", relu.activation.ReLU)
}

func (leakyReLU *LeakyReLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryMetalForward(stateDict, "metal.activation.leaky_relu", func(input []float64) ([]float64, error) {
		return leakyReLU.activation.LeakyReLU(input, stateDict.Alpha)
	})
}

func (gelu *GELU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryMetalForward(stateDict, "metal.activation.gelu", gelu.activation.GELU)
}

func (tanh *Tanh) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryMetalForward(stateDict, "metal.activation.tanh", tanh.activation.Tanh)
}

func (sigmoid *Sigmoid) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryMetalForward(stateDict, "metal.activation.sigmoid", sigmoid.activation.Sigmoid)
}

func (swiglu *SwiGLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryMetalForward(stateDict, "metal.activation.swiglu", swiglu.activation.SwiGLU)
}

func (swish *Swish) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryMetalForward(stateDict, "metal.activation.swish", swish.activation.Swish)
}

func (selu *SELU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryMetalForward(stateDict, "metal.activation.selu", selu.activation.SELU)
}

type SDPA struct{ attention *MetalAttention }
type MQA struct{ attention *MetalAttention }
type GQA struct{ attention *MetalAttention }
type SlidingWindowAttention struct{ attention *MetalAttention }

func (sdpa *SDPA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.attention.sdpa", 4, 3)

	if err != nil {
		return nil, err
	}

	output, err := sdpa.attention.SDPA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3], stateDict.Causal,
	)

	return setMetalOutput(stateDict, output, err)
}

func (mqa *MQA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.attention.mqa", 4, 3)

	if err != nil {
		return nil, err
	}

	output, err := mqa.attention.MQA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3],
	)

	return setMetalOutput(stateDict, output, err)
}

func (gqa *GQA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("metal.attention.gqa", 3); err != nil {
		return nil, err
	}

	batch, numHeads, numKVHeads, sequenceLength, headDim, err := stateDict.GQALayout(
		"metal.attention.gqa",
	)

	if err != nil {
		return nil, err
	}

	output, err := gqa.attention.GQA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		batch, numHeads, numKVHeads, sequenceLength, headDim, stateDict.Causal,
	)

	return setMetalOutput(stateDict, output, err)
}

func (attention *SlidingWindowAttention) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.attention.sliding_window", 4, 3)

	if err != nil {
		return nil, err
	}

	output, err := attention.attention.SlidingWindow(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3], stateDict.Window,
	)

	return setMetalOutput(stateDict, output, err)
}

type ApplyMask struct{ masking *MetalMasking }
type CausalMask struct{ masking *MetalMasking }

func (applyMask *ApplyMask) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("metal.masking.apply_mask", 2); err != nil {
		return nil, err
	}

	output, err := applyMask.masking.NewApplyMask().Forward(
		stateDict.OperationShape(), stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

func (causalMask *CausalMask) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.masking.causal_mask"); err != nil {
		return nil, err
	}

	output, err := causalMask.masking.NewCausalMask().Forward(
		stateDict.OperationShape(), stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

type Add struct{ mathOps *MathOps }
type Mul struct{ mathOps *MathOps }
type Matmul struct{ mathOps *MathOps }
type Exp struct{ mathOps *MathOps }
type Log struct{ mathOps *MathOps }
type LogSumExp struct{ mathOps *MathOps }
type Softmax struct{ mathOps *MathOps }
type Outer struct{ mathOps *MathOps }
type Sign struct{ mathOps *MathOps }
type InvSqrtDimScale struct{ mathOps *MathOps }
type Dropout struct{ mathOps *MathOps }
type RMSNorm struct{ mathOps *MathOps }
type LayerNorm struct{ mathOps *MathOps }
type GroupNorm struct{ mathOps *MathOps }

func (add *Add) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("metal.math.add", 2); err != nil {
		return nil, err
	}

	output, err := add.mathOps.Add(stateDict.OperationShape(), stateDict.Inputs...)
	return setMetalOutput(stateDict, output, err)
}

func (mul *Mul) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("metal.math.mul", 2); err != nil {
		return nil, err
	}

	output, err := mul.mathOps.Mul(stateDict.OperationShape(), stateDict.Inputs...)
	return setMetalOutput(stateDict, output, err)
}

func (matmul *Matmul) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("metal.math.matmul", 2); err != nil {
		return nil, err
	}

	output, err := matmul.mathOps.Matmul(stateDict.OperationShape(), stateDict.Inputs...)
	return setMetalOutput(stateDict, output, err)
}

func (exp *Exp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, "metal.math.exp", exp.mathOps.Exp)
}

func (log *Log) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, "metal.math.log", log.mathOps.Log)
}

func (logSumExp *LogSumExp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.math.logsumexp"); err != nil {
		return nil, err
	}

	dimSize := stateDict.OperationLastDim()

	if dimSize <= 0 {
		return nil, fmt.Errorf("metal.math.logsumexp: last dimension must be positive")
	}

	if len(stateDict.Inputs[0])%dimSize != 0 {
		return nil, fmt.Errorf("metal.math.logsumexp: input length must divide last dimension")
	}

	output, err := logSumExp.mathOps.LogSumExp(stateDict.OperationShape(), stateDict.Inputs...)
	return setMetalOutput(stateDict, output, err)
}

func (softmax *Softmax) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, "metal.math.softmax", softmax.mathOps.Softmax)
}

func (outer *Outer) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("metal.math.outer", 2); err != nil {
		return nil, err
	}

	output, err := outer.mathOps.Outer(stateDict.OperationShape(), stateDict.Inputs...)
	return setMetalOutput(stateDict, output, err)
}

func (sign *Sign) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, "metal.math.sign", sign.mathOps.Sign)
}

func (scale *InvSqrtDimScale) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryShapeMetalForward(
		stateDict, "metal.math.inv_sqrt_dim_scale", scale.mathOps.InvSqrtDimScale,
	)
}

func (dropout *Dropout) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.math.dropout"); err != nil {
		return nil, err
	}

	output, err := dropout.mathOps.Dropout(
		stateDict.P, stateDict.Training, stateDict.Step, stateDict.Inputs[0],
	)

	return setMetalOutput(stateDict, output, err)
}

func (rmsNorm *RMSNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.math.rmsnorm"); err != nil {
		return nil, err
	}

	output, err := rmsNorm.mathOps.RMSNorm(
		stateDict.OperationShape(), stateDict.Eps, stateDict.Weight, stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

func (layerNorm *LayerNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.math.layernorm"); err != nil {
		return nil, err
	}

	output, err := layerNorm.mathOps.LayerNorm(
		stateDict.OperationShape(), stateDict.Eps, stateDict.Weight, stateDict.Bias,
		stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

func (groupNorm *GroupNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.math.groupnorm"); err != nil {
		return nil, err
	}

	output, err := groupNorm.mathOps.GroupNorm(
		stateDict.OperationShape(),
		stateDict.Eps,
		stateDict.Groups,
		stateDict.Weight,
		stateDict.Bias,
		stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

type Reshape struct{ shapeOps *MetalShapeOps }
type Transpose struct{ shapeOps *MetalShapeOps }
type Concat struct{ shapeOps *MetalShapeOps }
type Split struct{ shapeOps *MetalShapeOps }
type UpsampleNearest2D struct{ shapeOps *MetalShapeOps }
type ViewAsHeads struct{ shapeOps *MetalShapeOps }
type MergeHeads struct{ shapeOps *MetalShapeOps }
type LastToken struct{ shapeOps *MetalShapeOps }

func (reshape *Reshape) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, "metal.shape.reshape", func(
		shape []int, data ...[]float64,
	) ([]float64, error) {
		return reshape.shapeOps.Copy(data[0])
	})
}

func (transpose *Transpose) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.shape.transpose"); err != nil {
		return nil, err
	}

	output, err := transpose.shapeOps.Transpose(
		stateDict.OperationShape(), stateDict.Dim0, stateDict.Dim1, stateDict.Inputs[0],
	)

	return setMetalOutput(stateDict, output, err)
}

func (concat *Concat) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("metal.shape.concat", 2); err != nil {
		return nil, err
	}

	output := append([]float64(nil), stateDict.Inputs[0]...)

	for _, input := range stateDict.Inputs[1:] {
		next, err := concat.shapeOps.Concat(output, input)

		if err != nil {
			return nil, err
		}

		output = next
	}

	stateDict.SetOperationOutput(output)

	return stateDict, nil
}

func (split *Split) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.shape.split"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()
	rank := len(shape)

	if stateDict.Dim < 0 || stateDict.Dim >= rank {
		return nil, fmt.Errorf("metal.shape.split: dim %d out of range rank %d", stateDict.Dim, rank)
	}

	if stateDict.SplitSize <= 0 {
		return nil, fmt.Errorf("metal.shape.split: split size must be positive")
	}

	outer := 1

	for dimension := 0; dimension < stateDict.Dim; dimension++ {
		outer *= shape[dimension]
	}

	inner := 1

	for dimension := stateDict.Dim + 1; dimension < rank; dimension++ {
		inner *= shape[dimension]
	}

	dimSize := shape[stateDict.Dim]

	if dimSize%stateDict.SplitSize != 0 {
		return nil, fmt.Errorf("metal.shape.split: dim size is not divisible by split size")
	}

	output, err := split.shapeOps.Split(
		stateDict.Inputs[0], outer, dimSize, stateDict.SplitSize, inner,
	)

	return setMetalOutput(stateDict, output, err)
}

func (upsample *UpsampleNearest2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.shape.upsample_nearest2d", 4, 1)

	if err != nil {
		return nil, err
	}

	scaleH, scaleW, err := metalUpsampleNearest2DScale(stateDict, shape)

	if err != nil {
		return nil, err
	}

	output, err := upsample.shapeOps.UpsampleNearest2D(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3], scaleH, scaleW,
	)

	return setMetalOutput(stateDict, output, err)
}

func (viewAsHeads *ViewAsHeads) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.shape.view_as_heads", 3, 1)

	if err != nil {
		return nil, err
	}

	if stateDict.NumHeads <= 0 {
		return nil, fmt.Errorf("metal.shape.view_as_heads: NumHeads must be > 0")
	}

	if shape[2]%stateDict.NumHeads != 0 {
		return nil, fmt.Errorf(
			"metal.shape.view_as_heads: model dimension %d not divisible by NumHeads %d",
			shape[2], stateDict.NumHeads,
		)
	}

	output, err := viewAsHeads.shapeOps.ViewAsHeads(
		stateDict.Inputs[0], shape[0], shape[1], stateDict.NumHeads,
		shape[2]/stateDict.NumHeads,
	)

	return setMetalOutput(stateDict, output, err)
}

func (mergeHeads *MergeHeads) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.shape.merge_heads", 4, 1)

	if err != nil {
		return nil, err
	}

	output, err := mergeHeads.shapeOps.MergeHeads(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
	)

	return setMetalOutput(stateDict, output, err)
}

func (lastToken *LastToken) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.shape.last_token"); err != nil {
		return nil, err
	}

	outer, sequenceLength, featureLength, err := lastTokenShapeParts(
		stateDict.OperationShape(),
	)

	if err != nil {
		return nil, err
	}

	output, err := lastToken.shapeOps.LastToken(
		stateDict.Inputs[0],
		outer,
		sequenceLength,
		featureLength,
	)

	return setMetalOutput(stateDict, output, err)
}

type RoPE struct{ positional *MetalPositional }
type ALiBi struct{ positional *MetalPositional }

func (rope *RoPE) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.positional.rope"); err != nil {
		return nil, err
	}

	batch, numHeads, sequenceLength, headDim, err := stateDict.RoPELayout(
		"metal.positional.rope",
	)

	if err != nil {
		return nil, err
	}

	output, err := rope.positional.RoPEForwardAtModeConfig(
		rotary.Config{
			Base:                          defaultFloat(stateDict.Base, 10000),
			Type:                          stateDict.RoPEType,
			Factor:                        stateDict.RoPEFactor,
			LowFreqFactor:                 stateDict.RoPELowFreqFactor,
			HighFreqFactor:                stateDict.RoPEHighFreqFactor,
			OriginalMaxPositionEmbeddings: stateDict.RoPEOriginalContext,
		},
		stateDict.PositionStart,
		stateDict.Mode,
		[]int{batch, numHeads, sequenceLength, headDim},
		stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

func (alibi *ALiBi) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.positional.alibi"); err != nil {
		return nil, err
	}

	output, err := alibi.positional.ALiBiForwardCausal(
		stateDict.OperationShape(),
		stateDict.Causal,
	)

	return setMetalOutput(stateDict, output, err)
}

type TokenEmbedding struct{ embeddingOps *EmbeddingOps }

func (tokenEmbedding *TokenEmbedding) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("metal.embedding.token_embedding", 2); err != nil {
		return nil, err
	}

	output, err := tokenEmbedding.embeddingOps.Forward(
		stateDict.OperationShape(), stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

type Conv1D struct{ convolution *ConvolutionOps }
type Conv2D struct{ convolution *ConvolutionOps }
type Conv3D struct{ convolution *ConvolutionOps }
type ConvTranspose2D struct{ convolution *ConvolutionOps }

func (conv *Conv1D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.convolution.conv1d", 3, 1)

	if err != nil {
		return nil, err
	}

	output, err := conv.convolution.Conv1d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], stateDict.Weight,
		stateDict.Bias, stateDict.OutChannels, stateDict.KernelSize,
		defaultInt(stateDict.Stride, 1), stateDict.Padding,
		defaultInt(stateDict.Dilation, 1), defaultInt(stateDict.Groups, 1),
		stateDict.OutW,
	)

	if err != nil {
		return nil, fmt.Errorf("metal.convolution.conv1d node %q: %w", stateDict.NodeID, err)
	}

	return setMetalOutput(stateDict, output, nil)
}

func (conv *Conv2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.convolution.conv2d", 4, 1)

	if err != nil {
		return nil, err
	}

	output, err := conv.convolution.Conv2d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultInt(stateDict.KernelH, stateDict.KernelSize),
		defaultInt(stateDict.KernelW, stateDict.KernelSize),
		defaultInt(defaultInt(stateDict.StrideH, stateDict.Stride), 1),
		defaultInt(defaultInt(stateDict.StrideW, stateDict.Stride), 1),
		stateDict.PadH, stateDict.PadW,
		defaultInt(defaultInt(stateDict.DilationH, stateDict.Dilation), 1),
		defaultInt(defaultInt(stateDict.DilationW, stateDict.Dilation), 1),
		defaultInt(stateDict.Groups, 1), stateDict.OutH, stateDict.OutW,
	)

	if err != nil {
		return nil, fmt.Errorf("metal.convolution.conv2d node %q: %w", stateDict.NodeID, err)
	}

	return setMetalOutput(stateDict, output, nil)
}

func (conv *Conv3D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.convolution.conv3d", 5, 1)

	if err != nil {
		return nil, err
	}

	output, err := conv.convolution.Conv3d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3], shape[4],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultInt(stateDict.KernelD, stateDict.KernelSize),
		defaultInt(stateDict.KernelH, stateDict.KernelSize),
		defaultInt(stateDict.KernelW, stateDict.KernelSize),
		defaultInt(defaultInt(stateDict.StrideD, stateDict.Stride), 1),
		defaultInt(defaultInt(stateDict.StrideH, stateDict.Stride), 1),
		defaultInt(defaultInt(stateDict.StrideW, stateDict.Stride), 1),
		stateDict.PadD, stateDict.PadH, stateDict.PadW,
		defaultInt(defaultInt(stateDict.DilationD, stateDict.Dilation), 1),
		defaultInt(defaultInt(stateDict.DilationH, stateDict.Dilation), 1),
		defaultInt(defaultInt(stateDict.DilationW, stateDict.Dilation), 1),
		defaultInt(stateDict.Groups, 1),
		stateDict.Dim0, stateDict.OutH, stateDict.OutW,
	)

	if err != nil {
		return nil, fmt.Errorf("metal.convolution.conv3d node %q: %w", stateDict.NodeID, err)
	}

	return setMetalOutput(stateDict, output, nil)
}

func (conv *ConvTranspose2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireMetalShape(stateDict, "metal.convolution.conv_transpose2d", 4, 1)

	if err != nil {
		return nil, err
	}

	output, err := conv.convolution.ConvTranspose2d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultInt(stateDict.KernelH, stateDict.KernelSize),
		defaultInt(stateDict.KernelW, stateDict.KernelSize),
		defaultInt(defaultInt(stateDict.StrideH, stateDict.Stride), 1),
		defaultInt(defaultInt(stateDict.StrideW, stateDict.Stride), 1),
		stateDict.PadH, stateDict.PadW,
		defaultInt(defaultInt(stateDict.DilationH, stateDict.Dilation), 1),
		defaultInt(defaultInt(stateDict.DilationW, stateDict.Dilation), 1),
		defaultInt(stateDict.Groups, 1), stateDict.OutH, stateDict.OutW,
	)

	if err != nil {
		return nil, fmt.Errorf(
			"metal.convolution.conv_transpose2d node %q: %w",
			stateDict.NodeID,
			err,
		)
	}

	return setMetalOutput(stateDict, output, nil)
}

type MaxPool2D struct{ pooling *PoolingOps }
type AvgPool2D struct{ pooling *PoolingOps }
type AdaptiveAvgPool2D struct{ pooling *PoolingOps }
type AdaptiveMaxPool2D struct{ pooling *PoolingOps }

func (pool *MaxPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return poolForward(stateDict, "metal.pooling.max_pool2d", func(
		shape []int, input []float64,
	) ([]float64, error) {
		return pool.pooling.MaxPool2d(shape, maxPoolParams(stateDict), input)
	})
}

func (pool *AvgPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return poolForward(stateDict, "metal.pooling.avg_pool2d", func(
		shape []int, input []float64,
	) ([]float64, error) {
		return pool.pooling.AvgPool2d(shape, avgPoolParams(stateDict), input)
	})
}

func (pool *AdaptiveAvgPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return poolForward(stateDict, "metal.pooling.adaptive_avg_pool2d", func(
		shape []int, input []float64,
	) ([]float64, error) {
		return pool.pooling.AdaptiveAvgPool2d(shape, stateDict.OutH, stateDict.OutW, input)
	})
}

func (pool *AdaptiveMaxPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return poolForward(stateDict, "metal.pooling.adaptive_max_pool2d", func(
		shape []int, input []float64,
	) ([]float64, error) {
		return pool.pooling.AdaptiveMaxPool2d(shape, stateDict.OutH, stateDict.OutW, input)
	})
}

type Linear struct{ projection *ProjectionOps }
type FusedQKV struct{ projection *ProjectionOps }

func (linear *Linear) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.projection.linear"); err != nil {
		return nil, err
	}

	weight := stateDict.Weight

	if len(weight) == 0 && len(stateDict.Inputs) > 1 {
		weight = stateDict.Inputs[1]
	}

	bias := stateDict.Bias

	if len(bias) == 0 && len(stateDict.Inputs) > 2 {
		bias = stateDict.Inputs[2]
	}

	output, err := linear.projection.Linear(
		stateDict.OperationShape(), weight, bias, stateDict.Inputs[0],
	)

	return setMetalOutput(stateDict, output, err)
}

func (fusedQKV *FusedQKV) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.projection.fused_qkv"); err != nil {
		return nil, err
	}

	weight := stateDict.Weight

	if len(weight) == 0 && len(stateDict.Inputs) > 1 {
		weight = stateDict.Inputs[1]
	}

	bias := stateDict.Bias

	if len(bias) == 0 && len(stateDict.Inputs) > 2 {
		bias = stateDict.Inputs[2]
	}

	output, err := fusedQKV.projection.FusedQKV(
		stateDict.OperationShape(), weight, bias, stateDict.Inputs[0],
	)

	return setMetalOutput(stateDict, output, err)
}

type HawkesIntensity struct{ hawkes *MetalHawkes }
type HawkesKernelMatrix struct{ hawkes *MetalHawkes }
type HawkesLogLikelihood struct{ hawkes *MetalHawkes }
type HawkesSimulate struct{ hawkes *MetalHawkes }

func (intensity *HawkesIntensity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return hawkesForward(stateDict, "metal.hawkes.intensity", intensity.hawkes.Intensity)
}

func (kernelMatrix *HawkesKernelMatrix) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return hawkesForward(stateDict, "metal.hawkes.kernel_matrix", kernelMatrix.hawkes.KernelMatrix)
}

func (logLikelihood *HawkesLogLikelihood) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return hawkesForward(stateDict, "metal.hawkes.log_likelihood", logLikelihood.hawkes.LogLikelihood)
}

func (simulate *HawkesSimulate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return hawkesForward(stateDict, "metal.hawkes.simulate", simulate.hawkes.Simulate)
}

type VSABind struct{ vsaOps *MetalVSAOps }
type VSABundle struct{ vsaOps *MetalVSAOps }
type VSASimilarity struct{ vsaOps *MetalVSAOps }
type VSAPermute struct{ vsaOps *MetalVSAOps }
type VSAInversePermute struct{ vsaOps *MetalVSAOps }

func (bind *VSABind) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return vsaForward(stateDict, "metal.vsa.bind", bind.vsaOps.Bind)
}

func (bundle *VSABundle) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return vsaForward(stateDict, "metal.vsa.bundle", bundle.vsaOps.Bundle)
}

func (similarity *VSASimilarity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return vsaForward(stateDict, "metal.vsa.similarity", similarity.vsaOps.Similarity)
}

func (permute *VSAPermute) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.vsa.permute"); err != nil {
		return nil, err
	}

	output, err := permute.vsaOps.Permute(
		stateDict.OperationShape(), stateDict.K, stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

func (inversePermute *VSAInversePermute) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("metal.vsa.inverse_permute"); err != nil {
		return nil, err
	}

	output, err := inversePermute.vsaOps.InversePermute(
		stateDict.OperationShape(), stateDict.K, stateDict.Inputs...,
	)

	return setMetalOutput(stateDict, output, err)
}

type BeliefUpdate struct{ activeInference *ActiveInferenceOps }
type ExpectedFreeEnergy struct{ activeInference *ActiveInferenceOps }
type FreeEnergy struct{ activeInference *ActiveInferenceOps }
type PrecisionWeight struct{ activeInference *ActiveInferenceOps }

func (beliefUpdate *BeliefUpdate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return activeInferenceForward(
		stateDict, "metal.active_inference.belief_update",
		beliefUpdate.activeInference.BeliefUpdate,
	)
}

func (expectedFreeEnergy *ExpectedFreeEnergy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return activeInferenceForward(
		stateDict, "metal.active_inference.expected_free_energy",
		expectedFreeEnergy.activeInference.ExpectedFreeEnergy,
	)
}

func (freeEnergy *FreeEnergy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return activeInferenceForward(
		stateDict, "metal.active_inference.free_energy", freeEnergy.activeInference.FreeEnergy,
	)
}

func (precisionWeight *PrecisionWeight) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return activeInferenceForward(
		stateDict, "metal.active_inference.precision_weight",
		precisionWeight.activeInference.PrecisionWeight,
	)
}

type Prediction struct{ predictiveCoding *MetalPredictiveCodingOps }
type PredictionError struct{ predictiveCoding *MetalPredictiveCodingOps }
type UpdateRepresentation struct{ predictiveCoding *MetalPredictiveCodingOps }
type UpdateWeights struct{ predictiveCoding *MetalPredictiveCodingOps }

func (prediction *Prediction) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return predictiveCodingForward(
		stateDict, "metal.predictive_coding.prediction",
		prediction.predictiveCoding.Prediction,
	)
}

func (predictionError *PredictionError) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return predictiveCodingForward(
		stateDict, "metal.predictive_coding.prediction_error",
		predictionError.predictiveCoding.PredictionError,
	)
}

func (updateRepresentation *UpdateRepresentation) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return predictiveCodingForward(
		stateDict, "metal.predictive_coding.update_representation",
		updateRepresentation.predictiveCoding.UpdateRepresentation,
	)
}

func (updateWeights *UpdateWeights) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return predictiveCodingForward(
		stateDict, "metal.predictive_coding.update_weights",
		updateWeights.predictiveCoding.UpdateWeights,
	)
}

type FlowActive struct{ markovBlanket *MetalMarkovBlanket }
type FlowInternal struct{ markovBlanket *MetalMarkovBlanket }
type MutualInformation struct{ markovBlanket *MetalMarkovBlanket }
type Partition struct{ markovBlanket *MetalMarkovBlanket }

func (flowActive *FlowActive) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return markovBlanketForward(
		stateDict, "metal.markov_blanket.flow_active",
		flowActive.markovBlanket.FlowActive,
	)
}

func (flowInternal *FlowInternal) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return markovBlanketForward(
		stateDict, "metal.markov_blanket.flow_internal",
		flowInternal.markovBlanket.FlowInternal,
	)
}

func (mutualInformation *MutualInformation) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return markovBlanketForward(
		stateDict, "metal.markov_blanket.mutual_information",
		mutualInformation.markovBlanket.MutualInformation,
	)
}

func (partition *Partition) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return markovBlanketForward(
		stateDict, "metal.markov_blanket.partition", partition.markovBlanket.Partition,
	)
}

type BackdoorAdjustment struct{ causal *MetalCausalOps }
type CATE struct{ causal *MetalCausalOps }
type Counterfactual struct{ causal *MetalCausalOps }
type DAGMarkovFactorization struct{ causal *MetalCausalOps }
type DoCalculus struct{ causal *MetalCausalOps }
type FrontdoorAdjustment struct{ causal *MetalCausalOps }
type IVEstimate struct{ causal *MetalCausalOps }

func (backdoorAdjustment *BackdoorAdjustment) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return causalForward(
		stateDict, "metal.causal.backdoor_adjustment",
		backdoorAdjustment.causal.BackdoorAdjustment,
	)
}

func (cate *CATE) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return causalForward(stateDict, "metal.causal.cate", cate.causal.CATE)
}

func (counterfactual *Counterfactual) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return causalForward(
		stateDict, "metal.causal.counterfactual", counterfactual.causal.Counterfactual,
	)
}

func (dagMarkovFactorization *DAGMarkovFactorization) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return causalForward(
		stateDict, "metal.causal.dag_markov_factorization",
		dagMarkovFactorization.causal.DAGMarkovFactorization,
	)
}

func (doCalculus *DoCalculus) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return causalForward(stateDict, "metal.causal.do_calculus", doCalculus.causal.DoCalculus)
}

func (frontdoorAdjustment *FrontdoorAdjustment) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return causalForward(
		stateDict, "metal.causal.frontdoor_adjustment",
		frontdoorAdjustment.causal.FrontdoorAdjustment,
	)
}

func (ivEstimate *IVEstimate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return causalForward(stateDict, "metal.causal.iv_estimate", ivEstimate.causal.IVEstimate)
}

type MSELoss struct{ trainingOps *MetalTrainingOps }
type CrossEntropyLoss struct{ trainingOps *MetalTrainingOps }
type MSEGrad struct{ trainingOps *MetalTrainingOps }
type CrossEntropyGrad struct{ trainingOps *MetalTrainingOps }
type Accuracy struct{ trainingOps *MetalTrainingOps }
type Perplexity struct{ trainingOps *MetalTrainingOps }
type F1 struct{ trainingOps *MetalTrainingOps }

func (mseLoss *MSELoss) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := metalTrainInputs(stateDict, "metal.train.mse_loss"); err != nil {
		return nil, err
	}

	output, err := mseLoss.trainingOps.MSELoss(stateDict.Inputs[0], stateDict.Inputs[1])
	return setMetalOutput(stateDict, output, err)
}

func (crossEntropyLoss *CrossEntropyLoss) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := metalTrainInputs(stateDict, "metal.train.cross_entropy_loss"); err != nil {
		return nil, err
	}

	output, err := crossEntropyLoss.trainingOps.CrossEntropyLoss(
		stateDict.Inputs[0], stateDict.Inputs[1],
	)

	return setMetalOutput(stateDict, output, err)
}

func (mseGrad *MSEGrad) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := metalTrainInputs(stateDict, "metal.train.mse_grad"); err != nil {
		return nil, err
	}

	output, err := mseGrad.trainingOps.MSEGrad(stateDict.Inputs[0], stateDict.Inputs[1])
	return setMetalOutput(stateDict, output, err)
}

func (crossEntropyGrad *CrossEntropyGrad) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := metalTrainInputs(stateDict, "metal.train.cross_entropy_grad"); err != nil {
		return nil, err
	}

	output, err := crossEntropyGrad.trainingOps.CrossEntropyGrad(
		stateDict.Inputs[0], stateDict.Inputs[1],
	)

	return setMetalOutput(stateDict, output, err)
}

func (accuracy *Accuracy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := metalTrainInputs(stateDict, "metal.bench.accuracy"); err != nil {
		return nil, err
	}

	output, err := accuracy.trainingOps.Accuracy(stateDict.Inputs[0], stateDict.Inputs[1])

	if err != nil {
		return nil, err
	}

	stateDict.Total++

	if stdmath.Abs(output[0]-1.0) < 1e-6 {
		stateDict.Correct++
	}
	stateDict.EnsureOperationOutLen(1)
	stateDict.Out[0] = float64(stateDict.Correct) / float64(stateDict.Total)

	return stateDict, nil
}

func (perplexity *Perplexity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := metalTrainInputs(stateDict, "metal.bench.perplexity"); err != nil {
		return nil, err
	}

	output, err := perplexity.trainingOps.CrossEntropyLoss(
		stateDict.Inputs[0], stateDict.Inputs[1],
	)

	if err != nil {
		return nil, err
	}

	stateDict.Total++
	stateDict.Sum += output[0]
	stateDict.EnsureOperationOutLen(1)
	stateDict.Out[0] = stdmath.Exp(stateDict.Sum / float64(stateDict.Total))

	return stateDict, nil
}

func (f1 *F1) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := metalTrainInputs(stateDict, "metal.bench.f1"); err != nil {
		return nil, err
	}

	counts, err := f1.trainingOps.F1Counts(stateDict.Inputs[0], stateDict.Inputs[1])

	if err != nil {
		return nil, err
	}

	stateDict.TP += counts[0]
	stateDict.FP += counts[1]
	stateDict.FN += counts[2]
	precision := stateDict.TP / (stateDict.TP + stateDict.FP + 1e-9)
	recall := stateDict.TP / (stateDict.TP + stateDict.FN + 1e-9)
	stateDict.EnsureOperationOutLen(1)
	stateDict.Out[0] = 2 * precision * recall / (precision + recall + 1e-9)

	return stateDict, nil
}

func newMetalActivation(config *state.Dict) (*MetalActivation, error) {
	return New(metalLibrary(config, "activation.metallib"))
}

func newMetalAttention(config *state.Dict) (*MetalAttention, error) {
	return NewAttention(metalLibrary(config, "attention.metallib"))
}

func newMetalMasking(config *state.Dict) (*MetalMasking, error) {
	return NewMasking(metalLibrary(config, "masking.metallib"))
}

func newMetalMath(config *state.Dict) (*MathOps, error) {
	return NewMathOps(metalLibrary(config, "math.metallib"))
}

func newMetalTraining(config *state.Dict) (*MetalTrainingOps, error) {
	return NewTrainingOps(metalLibrary(config, "math.metallib"))
}

func newMetalShape(config *state.Dict) (*MetalShapeOps, error) {
	return NewShapeOps(metalLibrary(config, "shape.metallib"))
}

func newMetalPositional(config *state.Dict) (*MetalPositional, error) {
	return NewPositional(metalLibrary(config, "positional.metallib"))
}

func newMetalConvolution(config *state.Dict) (*ConvolutionOps, error) {
	return NewConvolutionOps(metalLibrary(config, "convolution.metallib"))
}

func newMetalPooling(config *state.Dict) (*PoolingOps, error) {
	return NewPoolingOps(metalLibrary(config, "pooling.metallib"))
}

func newMetalProjection(config *state.Dict) (*ProjectionOps, error) {
	return NewProjectionOps(metalLibrary(config, "projection.metallib"))
}

func newMetalHawkes(config *state.Dict) (*MetalHawkes, error) {
	return NewHawkes(metalLibrary(config, "hawkes.metallib"))
}

func newMetalVSA(config *state.Dict) (*MetalVSAOps, error) {
	return NewVSAOps(metalLibrary(config, "vsa.metallib"))
}

func newMetalActiveInference(config *state.Dict) (*ActiveInferenceOps, error) {
	return NewActiveInferenceOps(metalLibrary(config, "active_inference.metallib"))
}

func newMetalPredictiveCoding(config *state.Dict) (*MetalPredictiveCodingOps, error) {
	return NewPredictiveCodingOps(metalLibrary(config, "predictive_coding.metallib"))
}

func newMetalMarkovBlanket(config *state.Dict) (*MetalMarkovBlanket, error) {
	return NewMarkovBlanket(metalLibrary(config, "markov_blanket.metallib"))
}

func newMetalCausal(config *state.Dict) (*MetalCausalOps, error) {
	return NewCausalOps(metalLibrary(config, "causal.metallib"))
}

func metalLibrary(config *state.Dict, name string) string {
	config = stateConfig(config)

	if config.File != "" {
		return config.File
	}

	_, currentFile, _, ok := runtime.Caller(0)

	if !ok {
		return name
	}

	return filepath.Join(filepath.Dir(currentFile), name)
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

func defaultFloat(value, fallback float64) float64 {
	if value == 0 {
		return fallback
	}

	return value
}

func requireMetalShape(
	stateDict *state.Dict, name string, rank int, inputCount int,
) ([]int, error) {
	if err := stateDict.RequireOperationInputs(name, inputCount); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < rank {
		return nil, fmt.Errorf("%s: len(shape)=%d, need >= %d", name, len(shape), rank)
	}

	return shape, nil
}

func unaryMetalForward(
	stateDict *state.Dict,
	name string,
	run func([]float64) ([]float64, error),
) (*state.Dict, error) {
	if err := stateDict.RequireOperation(name); err != nil {
		return nil, err
	}

	output, err := run(stateDict.Inputs[0])

	return setMetalOutput(stateDict, output, err)
}

func unaryShapeMetalForward(
	stateDict *state.Dict,
	name string,
	run func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	if err := stateDict.RequireOperation(name); err != nil {
		return nil, err
	}

	output, err := run(stateDict.OperationShape(), stateDict.Inputs...)

	return setMetalOutput(stateDict, output, err)
}

func poolForward(
	stateDict *state.Dict,
	name string,
	run func([]int, []float64) ([]float64, error),
) (*state.Dict, error) {
	if err := stateDict.RequireOperation(name); err != nil {
		return nil, err
	}

	output, err := run(stateDict.OperationShape(), stateDict.Inputs[0])

	return setMetalOutput(stateDict, output, err)
}

func hawkesForward(
	stateDict *state.Dict,
	name string,
	run func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, name, run)
}

func vsaForward(
	stateDict *state.Dict,
	name string,
	run func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, name, run)
}

func activeInferenceForward(
	stateDict *state.Dict,
	name string,
	run func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, name, run)
}

func predictiveCodingForward(
	stateDict *state.Dict,
	name string,
	run func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, name, run)
}

func markovBlanketForward(
	stateDict *state.Dict,
	name string,
	run func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, name, run)
}

func causalForward(
	stateDict *state.Dict,
	name string,
	run func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	return unaryShapeMetalForward(stateDict, name, run)
}

func metalTrainInputs(stateDict *state.Dict, name string) error {
	if err := stateDict.RequireOperationInputs(name, 2); err != nil {
		return err
	}

	if len(stateDict.Inputs[0]) == 0 || len(stateDict.Inputs[0]) != len(stateDict.Inputs[1]) {
		return fmt.Errorf("%s: input lengths must match and be non-zero", name)
	}

	return nil
}

func setMetalOutput(
	stateDict *state.Dict,
	output []float64,
	err error,
) (*state.Dict, error) {
	if err != nil {
		return nil, err
	}

	stateDict.SetOperationOutput(output)

	return stateDict, nil
}

func maxPoolParams(stateDict *state.Dict) MaxPool2dParams {
	return MaxPool2dParams{
		KernelH: defaultInt(stateDict.KernelH, stateDict.KernelSize),
		KernelW: defaultInt(stateDict.KernelW, stateDict.KernelSize),
		StrideH: defaultInt(stateDict.StrideH, stateDict.Stride),
		StrideW: defaultInt(stateDict.StrideW, stateDict.Stride),
		PadH:    stateDict.PadH, PadW: stateDict.PadW,
		DilationH: defaultInt(stateDict.DilationH, stateDict.Dilation),
		DilationW: defaultInt(stateDict.DilationW, stateDict.Dilation),
		CeilMode:  stateDict.Ceil,
	}
}

func avgPoolParams(stateDict *state.Dict) AvgPool2dParams {
	return AvgPool2dParams{
		KernelH: defaultInt(stateDict.KernelH, stateDict.KernelSize),
		KernelW: defaultInt(stateDict.KernelW, stateDict.KernelSize),
		StrideH: defaultInt(stateDict.StrideH, stateDict.Stride),
		StrideW: defaultInt(stateDict.StrideW, stateDict.Stride),
		PadH:    stateDict.PadH, PadW: stateDict.PadW,
		DilationH:       defaultInt(stateDict.DilationH, stateDict.Dilation),
		DilationW:       defaultInt(stateDict.DilationW, stateDict.Dilation),
		CeilMode:        stateDict.Ceil,
		CountIncludePad: stateDict.CountPad,
		DivisorOverride: stateDict.Divisor,
	}
}
