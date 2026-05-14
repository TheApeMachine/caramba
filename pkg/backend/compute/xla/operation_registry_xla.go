//go:build cgo && xla

package xla

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func (registry OperationRegistry) ReLU(config *state.Dict) (state.Operation, error) {
	activation, err := newXLAActivation(config)
	if err != nil {
		return nil, err
	}

	return &ReLU{activation: activation}, nil
}

func (registry OperationRegistry) LeakyReLU(config *state.Dict) (state.Operation, error) {
	activation, err := newXLAActivation(config)
	if err != nil {
		return nil, err
	}

	return &LeakyReLU{activation: activation}, nil
}

func (registry OperationRegistry) GELU(config *state.Dict) (state.Operation, error) {
	activation, err := newXLAActivation(config)
	if err != nil {
		return nil, err
	}

	return &GELU{activation: activation}, nil
}

func (registry OperationRegistry) Tanh(config *state.Dict) (state.Operation, error) {
	activation, err := newXLAActivation(config)
	if err != nil {
		return nil, err
	}

	return &Tanh{activation: activation}, nil
}

func (registry OperationRegistry) Sigmoid(config *state.Dict) (state.Operation, error) {
	activation, err := newXLAActivation(config)
	if err != nil {
		return nil, err
	}

	return &Sigmoid{activation: activation}, nil
}

func (registry OperationRegistry) SwiGLU(config *state.Dict) (state.Operation, error) {
	activation, err := newXLAActivation(config)
	if err != nil {
		return nil, err
	}

	return &SwiGLU{activation: activation}, nil
}

func (registry OperationRegistry) Swish(config *state.Dict) (state.Operation, error) {
	activation, err := newXLAActivation(config)
	if err != nil {
		return nil, err
	}

	return &Swish{activation: activation}, nil
}

func (registry OperationRegistry) SDPA(config *state.Dict) (state.Operation, error) {
	attention, err := NewAttention(xlaPlatform(config))
	return &SDPA{attention: attention}, err
}

func (registry OperationRegistry) MQA(config *state.Dict) (state.Operation, error) {
	attention, err := NewAttention(xlaPlatform(config))
	return &MQA{attention: attention}, err
}

func (registry OperationRegistry) GQA(config *state.Dict) (state.Operation, error) {
	attention, err := NewAttention(xlaPlatform(config))
	return &GQA{attention: attention}, err
}

func (registry OperationRegistry) SlidingWindowAttention(config *state.Dict) (state.Operation, error) {
	attention, err := NewAttention(xlaPlatform(config))
	return &SlidingWindowAttention{attention: attention}, err
}

func (registry OperationRegistry) ApplyMask(config *state.Dict) (state.Operation, error) {
	masking, err := NewMasking(xlaPlatform(config))
	return &ApplyMask{masking: masking}, err
}

func (registry OperationRegistry) CausalMask(config *state.Dict) (state.Operation, error) {
	masking, err := NewMasking(xlaPlatform(config))
	return &CausalMask{masking: masking}, err
}

func (registry OperationRegistry) Add(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Add{mathOps: mathOps}, err
}

func (registry OperationRegistry) Mul(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Mul{mathOps: mathOps}, err
}

func (registry OperationRegistry) Matmul(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Matmul{mathOps: mathOps}, err
}

func (registry OperationRegistry) Exp(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Exp{mathOps: mathOps}, err
}

func (registry OperationRegistry) Log(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Log{mathOps: mathOps}, err
}

func (registry OperationRegistry) LogSumExp(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &LogSumExp{mathOps: mathOps}, err
}

func (registry OperationRegistry) Softmax(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Softmax{mathOps: mathOps}, err
}

func (registry OperationRegistry) Outer(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Outer{mathOps: mathOps}, err
}

func (registry OperationRegistry) Sign(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Sign{mathOps: mathOps}, err
}

func (registry OperationRegistry) InvSqrtDimScale(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &InvSqrtDimScale{mathOps: mathOps}, err
}

func (registry OperationRegistry) Dropout(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &Dropout{mathOps: mathOps}, err
}

func (registry OperationRegistry) RMSNorm(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &RMSNorm{mathOps: mathOps}, err
}

func (registry OperationRegistry) LayerNorm(config *state.Dict) (state.Operation, error) {
	mathOps, err := NewMathOps(xlaPlatform(config))
	return &LayerNorm{mathOps: mathOps}, err
}

func (registry OperationRegistry) Reshape(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newXLAShapeOps(config)
	return &Reshape{shapeOps: shapeOps}, err
}

func (registry OperationRegistry) Transpose(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newXLAShapeOps(config)
	return &Transpose{shapeOps: shapeOps}, err
}

func (registry OperationRegistry) Concat(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newXLAShapeOps(config)
	return &Concat{shapeOps: shapeOps}, err
}

func (registry OperationRegistry) Split(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newXLAShapeOps(config)
	return &Split{shapeOps: shapeOps}, err
}

func (registry OperationRegistry) ViewAsHeads(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newXLAShapeOps(config)
	return &ViewAsHeads{shapeOps: shapeOps}, err
}

func (registry OperationRegistry) MergeHeads(config *state.Dict) (state.Operation, error) {
	shapeOps, err := newXLAShapeOps(config)
	return &MergeHeads{shapeOps: shapeOps}, err
}

func (registry OperationRegistry) RoPE(config *state.Dict) (state.Operation, error) {
	positional, err := NewPositional(xlaPlatform(config))
	return &RoPE{positional: positional}, err
}

func (registry OperationRegistry) ALiBi(config *state.Dict) (state.Operation, error) {
	positional, err := NewPositional(xlaPlatform(config))
	return &ALiBi{positional: positional}, err
}

func (registry OperationRegistry) TokenEmbedding(config *state.Dict) (state.Operation, error) {
	config = xlaStateConfig(config)
	embedding, err := NewXLAEmbedding(xlaPlatform(config), config.VocabSize, config.DModel)
	return &TokenEmbedding{embedding: embedding}, err
}

func (registry OperationRegistry) Conv1D(config *state.Dict) (state.Operation, error) {
	convolution, err := NewXLAConvolution(xlaPlatform(config))
	return &Conv1D{convolution: convolution}, err
}

func (registry OperationRegistry) Conv2D(config *state.Dict) (state.Operation, error) {
	convolution, err := NewXLAConvolution(xlaPlatform(config))
	return &Conv2D{convolution: convolution}, err
}

func (registry OperationRegistry) Conv3D(config *state.Dict) (state.Operation, error) {
	convolution, err := NewXLAConvolution(xlaPlatform(config))
	return &Conv3D{convolution: convolution}, err
}

func (registry OperationRegistry) ConvTranspose2D(config *state.Dict) (state.Operation, error) {
	convolution, err := NewXLAConvolution(xlaPlatform(config))
	return &ConvTranspose2D{convolution: convolution}, err
}

func (registry OperationRegistry) MaxPool2D(config *state.Dict) (state.Operation, error) {
	pooling, err := NewXLAPooling(xlaPlatform(config))
	return &MaxPool2D{pooling: pooling}, err
}

func (registry OperationRegistry) AvgPool2D(config *state.Dict) (state.Operation, error) {
	pooling, err := NewXLAPooling(xlaPlatform(config))
	return &AvgPool2D{pooling: pooling}, err
}

func (registry OperationRegistry) AdaptiveAvgPool2D(config *state.Dict) (state.Operation, error) {
	pooling, err := NewXLAPooling(xlaPlatform(config))
	return &AdaptiveAvgPool2D{pooling: pooling}, err
}

func (registry OperationRegistry) AdaptiveMaxPool2D(config *state.Dict) (state.Operation, error) {
	pooling, err := NewXLAPooling(xlaPlatform(config))
	return &AdaptiveMaxPool2D{pooling: pooling}, err
}

func (registry OperationRegistry) Linear(config *state.Dict) (state.Operation, error) {
	projection, err := NewXLAProjection(xlaPlatform(config))
	return &Linear{projection: projection}, err
}

func (registry OperationRegistry) FusedQKV(config *state.Dict) (state.Operation, error) {
	projection, err := NewXLAProjection(xlaPlatform(config))
	return &FusedQKV{projection: projection}, err
}

func (registry OperationRegistry) HawkesIntensity(config *state.Dict) (state.Operation, error) {
	hawkes, err := NewHawkes(xlaPlatform(config))
	return &HawkesIntensity{hawkes: hawkes}, err
}

func (registry OperationRegistry) HawkesKernelMatrix(config *state.Dict) (state.Operation, error) {
	hawkes, err := NewHawkes(xlaPlatform(config))
	return &HawkesKernelMatrix{hawkes: hawkes}, err
}

func (registry OperationRegistry) HawkesLogLikelihood(config *state.Dict) (state.Operation, error) {
	hawkes, err := NewHawkes(xlaPlatform(config))
	return &HawkesLogLikelihood{hawkes: hawkes}, err
}

func (registry OperationRegistry) HawkesSimulate(config *state.Dict) (state.Operation, error) {
	hawkes, err := NewHawkes(xlaPlatform(config))
	return &HawkesSimulate{hawkes: hawkes}, err
}

func (registry OperationRegistry) VSABind(config *state.Dict) (state.Operation, error) {
	vsaOps, err := NewVSAOps(xlaPlatform(config))
	return &VSABind{vsaOps: vsaOps}, err
}

func (registry OperationRegistry) VSABundle(config *state.Dict) (state.Operation, error) {
	vsaOps, err := NewVSAOps(xlaPlatform(config))
	return &VSABundle{vsaOps: vsaOps}, err
}

func (registry OperationRegistry) VSASimilarity(config *state.Dict) (state.Operation, error) {
	vsaOps, err := NewVSAOps(xlaPlatform(config))
	return &VSASimilarity{vsaOps: vsaOps}, err
}

func (registry OperationRegistry) VSAPermute(config *state.Dict) (state.Operation, error) {
	vsaOps, err := NewVSAOps(xlaPlatform(config))
	return &VSAPermute{vsaOps: vsaOps}, err
}

func (registry OperationRegistry) VSAInversePermute(config *state.Dict) (state.Operation, error) {
	vsaOps, err := NewVSAOps(xlaPlatform(config))
	return &VSAInversePermute{vsaOps: vsaOps}, err
}

func (registry OperationRegistry) BeliefUpdate(config *state.Dict) (state.Operation, error) {
	activeInference, err := NewActiveInferenceOps(xlaPlatform(config))
	return &BeliefUpdate{activeInference: activeInference}, err
}

func (registry OperationRegistry) ExpectedFreeEnergy(config *state.Dict) (state.Operation, error) {
	activeInference, err := NewActiveInferenceOps(xlaPlatform(config))
	return &ExpectedFreeEnergy{activeInference: activeInference}, err
}

func (registry OperationRegistry) FreeEnergy(config *state.Dict) (state.Operation, error) {
	activeInference, err := NewActiveInferenceOps(xlaPlatform(config))
	return &FreeEnergy{activeInference: activeInference}, err
}

func (registry OperationRegistry) PrecisionWeight(config *state.Dict) (state.Operation, error) {
	activeInference, err := NewActiveInferenceOps(xlaPlatform(config))
	return &PrecisionWeight{activeInference: activeInference}, err
}

func (registry OperationRegistry) Prediction(config *state.Dict) (state.Operation, error) {
	predictiveCoding, err := NewPredictiveCodingOps(xlaPlatform(config))
	return &Prediction{predictiveCoding: predictiveCoding}, err
}

func (registry OperationRegistry) PredictionError(config *state.Dict) (state.Operation, error) {
	predictiveCoding, err := NewPredictiveCodingOps(xlaPlatform(config))
	return &PredictionError{predictiveCoding: predictiveCoding}, err
}

func (registry OperationRegistry) UpdateRepresentation(config *state.Dict) (state.Operation, error) {
	predictiveCoding, err := NewPredictiveCodingOps(xlaPlatform(config))
	return &UpdateRepresentation{predictiveCoding: predictiveCoding}, err
}

func (registry OperationRegistry) UpdateWeights(config *state.Dict) (state.Operation, error) {
	predictiveCoding, err := NewPredictiveCodingOps(xlaPlatform(config))
	return &UpdateWeights{predictiveCoding: predictiveCoding}, err
}

func (registry OperationRegistry) FlowActive(config *state.Dict) (state.Operation, error) {
	markovBlanket, err := NewMarkovBlanket(xlaPlatform(config))
	return &FlowActive{markovBlanket: markovBlanket}, err
}

func (registry OperationRegistry) FlowInternal(config *state.Dict) (state.Operation, error) {
	markovBlanket, err := NewMarkovBlanket(xlaPlatform(config))
	return &FlowInternal{markovBlanket: markovBlanket}, err
}

func (registry OperationRegistry) MutualInformation(config *state.Dict) (state.Operation, error) {
	markovBlanket, err := NewMarkovBlanket(xlaPlatform(config))
	return &MutualInformation{markovBlanket: markovBlanket}, err
}

func (registry OperationRegistry) Partition(config *state.Dict) (state.Operation, error) {
	markovBlanket, err := NewMarkovBlanket(xlaPlatform(config))
	return &Partition{markovBlanket: markovBlanket}, err
}

func (registry OperationRegistry) BackdoorAdjustment(config *state.Dict) (state.Operation, error) {
	causal, err := NewCausalOps(xlaPlatform(config))
	return &BackdoorAdjustment{causal: causal}, err
}

func (registry OperationRegistry) CATE(config *state.Dict) (state.Operation, error) {
	causal, err := NewCausalOps(xlaPlatform(config))
	return &CATE{causal: causal}, err
}

func (registry OperationRegistry) Counterfactual(config *state.Dict) (state.Operation, error) {
	causal, err := NewCausalOps(xlaPlatform(config))
	return &Counterfactual{causal: causal}, err
}

func (registry OperationRegistry) DAGMarkovFactorization(
	config *state.Dict,
) (state.Operation, error) {
	causal, err := NewCausalOps(xlaPlatform(config))
	return &DAGMarkovFactorization{causal: causal}, err
}

func (registry OperationRegistry) DoCalculus(config *state.Dict) (state.Operation, error) {
	causal, err := NewCausalOps(xlaPlatform(config))
	return &DoCalculus{causal: causal}, err
}

func (registry OperationRegistry) FrontdoorAdjustment(config *state.Dict) (state.Operation, error) {
	causal, err := NewCausalOps(xlaPlatform(config))
	return &FrontdoorAdjustment{causal: causal}, err
}

func (registry OperationRegistry) IVEstimate(config *state.Dict) (state.Operation, error) {
	causal, err := NewCausalOps(xlaPlatform(config))
	return &IVEstimate{causal: causal}, err
}

func (registry OperationRegistry) MSELoss(config *state.Dict) (state.Operation, error) {
	trainingOps, err := NewTrainingOps(xlaPlatform(config))
	return &MSELoss{trainingOps: trainingOps}, err
}

func (registry OperationRegistry) CrossEntropyLoss(config *state.Dict) (state.Operation, error) {
	trainingOps, err := NewTrainingOps(xlaPlatform(config))
	return &CrossEntropyLoss{trainingOps: trainingOps}, err
}

func (registry OperationRegistry) MSEGrad(config *state.Dict) (state.Operation, error) {
	trainingOps, err := NewTrainingOps(xlaPlatform(config))
	return &MSEGrad{trainingOps: trainingOps}, err
}

func (registry OperationRegistry) CrossEntropyGrad(config *state.Dict) (state.Operation, error) {
	trainingOps, err := NewTrainingOps(xlaPlatform(config))
	return &CrossEntropyGrad{trainingOps: trainingOps}, err
}

func (registry OperationRegistry) Accuracy(config *state.Dict) (state.Operation, error) {
	trainingOps, err := NewTrainingOps(xlaPlatform(config))
	return &Accuracy{trainingOps: trainingOps}, err
}

func (registry OperationRegistry) Perplexity(config *state.Dict) (state.Operation, error) {
	trainingOps, err := NewTrainingOps(xlaPlatform(config))
	return &Perplexity{trainingOps: trainingOps}, err
}

func (registry OperationRegistry) F1(config *state.Dict) (state.Operation, error) {
	trainingOps, err := NewTrainingOps(xlaPlatform(config))
	return &F1{trainingOps: trainingOps}, err
}

func (registry OperationRegistry) Load(config *state.Dict) (state.Operation, error) {
	config = xlaStateConfig(config)
	return NewXLAModelOps().NewLoader(config.Source, config.File, config.Cache), nil
}

func (registry OperationRegistry) Surgery(config *state.Dict) (state.Operation, error) {
	config = xlaStateConfig(config)
	return NewXLAModelOps().NewSurgery(
		config.Source, config.Op, config.At, config.After, config.Name, config.Layer,
	), nil
}

func (registry OperationRegistry) Graft(config *state.Dict) (state.Operation, error) {
	config = xlaStateConfig(config)
	return NewXLAModelOps().NewGraft(config.Source, config.At, config.Mode), nil
}

func (registry OperationRegistry) LoRA(config *state.Dict) (state.Operation, error) {
	config = xlaStateConfig(config)
	return NewXLAModelOps().NewLoRA(
		config.Source, config.Preset, config.Targets, config.Rank, config.Alpha,
	), nil
}

func (registry OperationRegistry) Adapter(config *state.Dict) (state.Operation, error) {
	config = xlaStateConfig(config)
	return NewXLAModelOps().NewAdapter(config.Source, config.At, config.Reduction), nil
}

func (registry OperationRegistry) Freeze(config *state.Dict) (state.Operation, error) {
	config = xlaStateConfig(config)
	return NewXLAModelOps().NewFreeze(
		config.Source, config.Pattern, config.Except, config.Frozen,
	), nil
}

type ReLU struct{ activation *XLAActivation }
type LeakyReLU struct{ activation *XLAActivation }
type GELU struct{ activation *XLAActivation }
type Tanh struct{ activation *XLAActivation }
type Sigmoid struct{ activation *XLAActivation }
type Swish struct{ activation *XLAActivation }
type SwiGLU struct{ activation *XLAActivation }

func (relu *ReLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaUnaryForward(stateDict, "xla.activation.relu", relu.activation.ReLU)
}

func (leakyReLU *LeakyReLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaUnaryForward(
		stateDict,
		"xla.activation.leaky_relu",
		func(input []float64) ([]float64, error) {
			return leakyReLU.activation.LeakyReLU(input, stateDict.Alpha)
		},
	)
}

func (gelu *GELU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaUnaryForward(stateDict, "xla.activation.gelu", gelu.activation.GELU)
}

func (tanh *Tanh) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaUnaryForward(stateDict, "xla.activation.tanh", tanh.activation.Tanh)
}

func (sigmoid *Sigmoid) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaUnaryForward(stateDict, "xla.activation.sigmoid", sigmoid.activation.Sigmoid)
}

func (swish *Swish) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaUnaryForward(stateDict, "xla.activation.swish", swish.activation.Swish)
}

func (swiglu *SwiGLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaUnaryForward(stateDict, "xla.activation.swiglu", swiglu.activation.SwiGLU)
}

type SDPA struct{ attention *XLAAttention }
type MQA struct{ attention *XLAAttention }
type GQA struct{ attention *XLAAttention }
type SlidingWindowAttention struct{ attention *XLAAttention }

func (sdpa *SDPA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(stateDict, "xla.attention.sdpa", 4, 3)

	if err != nil {
		return nil, err
	}

	output, err := sdpa.attention.SDPA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3],
	)

	return setXLAOutput(stateDict, output, err)
}

func (mqa *MQA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(stateDict, "xla.attention.mqa", 4, 3)

	if err != nil {
		return nil, err
	}

	output, err := mqa.attention.MQA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3],
	)

	return setXLAOutput(stateDict, output, err)
}

func (gqa *GQA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(stateDict, "xla.attention.gqa", 5, 3)

	if err != nil {
		return nil, err
	}

	output, err := gqa.attention.GQA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3], shape[4],
	)

	return setXLAOutput(stateDict, output, err)
}

func (attention *SlidingWindowAttention) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(
		stateDict, "xla.attention.sliding_window_attention", 4, 3,
	)

	if err != nil {
		return nil, err
	}

	output, err := attention.attention.SlidingWindow(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3], stateDict.Window,
	)

	return setXLAOutput(stateDict, output, err)
}

type ApplyMask struct{ masking *XLAMasking }
type CausalMask struct{ masking *XLAMasking }

func (applyMask *ApplyMask) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("xla.masking.apply_mask", 2); err != nil {
		return nil, err
	}

	output, err := applyMask.masking.ApplyMask(stateDict.Inputs[0], stateDict.Inputs[1])
	return setXLAOutput(stateDict, output, err)
}

func (causalMask *CausalMask) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.masking.causal_mask"); err != nil {
		return nil, err
	}

	dim := stateDict.OperationLastDim()

	if dim <= 0 {
		return nil, fmt.Errorf("xla.masking.causal_mask: last dimension must be positive, got %d", dim)
	}

	output, err := causalMask.masking.CausalMask(dim)
	return setXLAOutput(stateDict, output, err)
}

type Add struct{ mathOps *XLAMathOps }
type Mul struct{ mathOps *XLAMathOps }
type Matmul struct{ mathOps *XLAMathOps }
type Exp struct{ mathOps *XLAMathOps }
type Log struct{ mathOps *XLAMathOps }
type LogSumExp struct{ mathOps *XLAMathOps }
type Softmax struct{ mathOps *XLAMathOps }
type Outer struct{ mathOps *XLAMathOps }
type Sign struct{ mathOps *XLAMathOps }
type InvSqrtDimScale struct{ mathOps *XLAMathOps }
type Dropout struct{ mathOps *XLAMathOps }
type RMSNorm struct{ mathOps *XLAMathOps }
type LayerNorm struct{ mathOps *XLAMathOps }

func (add *Add) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.math.add", 2, add.mathOps.Add)
}

func (mul *Mul) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.math.mul", 2, mul.mathOps.Mul)
}

func (matmul *Matmul) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.math.matmul", 2, matmul.mathOps.Matmul)
}

func (exp *Exp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.math.exp", 1, exp.mathOps.Exp)
}

func (log *Log) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.math.log", 1, log.mathOps.Log)
}

func (logSumExp *LogSumExp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.math.logsumexp"); err != nil {
		return nil, err
	}

	dimSize := stateDict.OperationLastDim()

	if dimSize <= 0 {
		return nil, fmt.Errorf("xla.math.logsumexp: last dimension must be positive")
	}

	if len(stateDict.Inputs[0])%dimSize != 0 {
		return nil, fmt.Errorf("xla.math.logsumexp: input length must be divisible by last dimension")
	}

	output, err := logSumExp.mathOps.LogSumExp(stateDict.OperationShape(), stateDict.Inputs...)

	return setXLAOutput(stateDict, output, err)
}

func (softmax *Softmax) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.math.softmax", 1, softmax.mathOps.Softmax)
}

func (outer *Outer) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.math.outer", 2, outer.mathOps.Outer)
}

func (sign *Sign) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.math.sign", 1, sign.mathOps.Sign)
}

func (scale *InvSqrtDimScale) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.math.inv_sqrt_dim_scale", 1, scale.mathOps.InvSqrtDimScale,
	)
}

func (dropout *Dropout) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.math.dropout"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.math.dropout: input[0] is required")
	}

	output, err := dropout.mathOps.Dropout(
		stateDict.P, stateDict.Training, stateDict.Step, stateDict.Inputs[0],
	)

	return setXLAOutput(stateDict, output, err)
}

func (rmsNorm *RMSNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.math.rmsnorm"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.math.rmsnorm: input[0] is required")
	}

	output, err := rmsNorm.mathOps.RMSNorm(
		stateDict.OperationShape(), stateDict.Eps, stateDict.Weight, stateDict.Inputs...,
	)

	return setXLAOutput(stateDict, output, err)
}

func (layerNorm *LayerNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.math.layernorm"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.math.layernorm: input[0] is required")
	}

	output, err := layerNorm.mathOps.LayerNorm(
		stateDict.OperationShape(), stateDict.Eps, stateDict.Weight, stateDict.Bias,
		stateDict.Inputs...,
	)

	return setXLAOutput(stateDict, output, err)
}

type Reshape struct{ shapeOps *XLAShapeOps }
type Transpose struct{ shapeOps *XLAShapeOps }
type Concat struct{ shapeOps *XLAShapeOps }
type Split struct{ shapeOps *XLAShapeOps }
type ViewAsHeads struct{ shapeOps *XLAShapeOps }
type MergeHeads struct{ shapeOps *XLAShapeOps }

func (reshape *Reshape) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.shape.reshape", 1, func(
		_ []int, data ...[]float64,
	) ([]float64, error) {
		return reshape.shapeOps.Copy(data[0])
	})
}

func (transpose *Transpose) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.shape.transpose"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.shape.transpose: input[0] is required")
	}

	output, err := transpose.shapeOps.Transpose(
		stateDict.OperationShape(), stateDict.Dim0, stateDict.Dim1, stateDict.Inputs[0],
	)

	return setXLAOutput(stateDict, output, err)
}

func (concat *Concat) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("xla.shape.concat", 2); err != nil {
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
	if err := stateDict.RequireOperation("xla.shape.split"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()
	rank := len(shape)

	if stateDict.Dim < 0 || stateDict.Dim >= rank {
		return nil, fmt.Errorf("xla.shape.split: dim %d out of range rank %d", stateDict.Dim, rank)
	}

	if stateDict.SplitSize <= 0 {
		return nil, fmt.Errorf("xla.shape.split: split size must be positive")
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
		return nil, fmt.Errorf("xla.shape.split: dim size is not divisible by split size")
	}

	output, err := split.shapeOps.Split(
		stateDict.Inputs[0], outer, dimSize, stateDict.SplitSize, inner,
	)

	return setXLAOutput(stateDict, output, err)
}

func (viewAsHeads *ViewAsHeads) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(stateDict, "xla.shape.view_as_heads", 3, 1)

	if err != nil {
		return nil, err
	}

	if stateDict.NumHeads <= 0 {
		return nil, fmt.Errorf("xla.shape.view_as_heads: num_heads must be positive")
	}

	if shape[2]%stateDict.NumHeads != 0 {
		return nil, fmt.Errorf("xla.shape.view_as_heads: hidden size must be divisible by num_heads")
	}

	output, err := viewAsHeads.shapeOps.ViewAsHeads(
		stateDict.Inputs[0], shape[0], shape[1], stateDict.NumHeads,
		shape[2]/stateDict.NumHeads,
	)

	return setXLAOutput(stateDict, output, err)
}

func (mergeHeads *MergeHeads) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(stateDict, "xla.shape.merge_heads", 4, 1)

	if err != nil {
		return nil, err
	}

	output, err := mergeHeads.shapeOps.MergeHeads(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
	)

	return setXLAOutput(stateDict, output, err)
}

type RoPE struct{ positional *XLAPositionalOps }
type ALiBi struct{ positional *XLAPositionalOps }

func (rope *RoPE) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.positional.rope"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.positional.rope: input[0] is required")
	}

	output, err := rope.positional.RoPEForward(
		defaultXLAFloat(stateDict.Base, 10000), stateDict.OperationShape(), stateDict.Inputs...,
	)

	return setXLAOutput(stateDict, output, err)
}

func (alibi *ALiBi) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.positional.alibi"); err != nil {
		return nil, err
	}

	output, err := alibi.positional.ALiBiForward(stateDict.OperationShape(), stateDict.Causal)
	return setXLAOutput(stateDict, output, err)
}

type TokenEmbedding struct{ embedding *XLAEmbedding }

func (tokenEmbedding *TokenEmbedding) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("xla.embedding.token_embedding", 2); err != nil {
		return nil, err
	}

	output, err := tokenEmbedding.embedding.Forward(stateDict.OperationShape(), stateDict.Inputs...)
	return setXLAOutput(stateDict, output, err)
}

type Conv1D struct{ convolution *XLAConvolution }
type Conv2D struct{ convolution *XLAConvolution }
type Conv3D struct{ convolution *XLAConvolution }
type ConvTranspose2D struct{ convolution *XLAConvolution }

func (convolution *Conv1D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(stateDict, "xla.convolution.conv1d", 3, 1)

	if err != nil {
		return nil, err
	}

	output, err := convolution.convolution.Conv1d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], stateDict.Weight,
		stateDict.Bias, stateDict.OutChannels, stateDict.KernelSize,
		defaultXLAInt(stateDict.Stride, 1), stateDict.Padding,
		defaultXLAInt(stateDict.Dilation, 1), defaultXLAInt(stateDict.Groups, 1),
		stateDict.OutW,
	)

	return setXLAOutput(stateDict, output, err)
}

func (convolution *Conv2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(stateDict, "xla.convolution.conv2d", 4, 1)

	if err != nil {
		return nil, err
	}

	output, err := convolution.convolution.Conv2d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultXLAInt(stateDict.KernelH, stateDict.KernelSize),
		defaultXLAInt(stateDict.KernelW, stateDict.KernelSize),
		defaultXLAInt(stateDict.StrideH, stateDict.Stride),
		defaultXLAInt(stateDict.StrideW, stateDict.Stride),
		stateDict.PadH, stateDict.PadW,
		defaultXLAInt(stateDict.DilationH, stateDict.Dilation),
		defaultXLAInt(stateDict.DilationW, stateDict.Dilation),
		defaultXLAInt(stateDict.Groups, 1), stateDict.OutH, stateDict.OutW,
	)

	return setXLAOutput(stateDict, output, err)
}

func (convolution *Conv3D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(stateDict, "xla.convolution.conv3d", 5, 1)

	if err != nil {
		return nil, err
	}

	output, err := convolution.convolution.Conv3d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3], shape[4],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultXLAInt(stateDict.KernelD, stateDict.KernelSize),
		defaultXLAInt(stateDict.KernelH, stateDict.KernelSize),
		defaultXLAInt(stateDict.KernelW, stateDict.KernelSize),
		defaultXLAInt(stateDict.StrideD, stateDict.Stride),
		defaultXLAInt(stateDict.StrideH, stateDict.Stride),
		defaultXLAInt(stateDict.StrideW, stateDict.Stride),
		stateDict.PadD, stateDict.PadH, stateDict.PadW,
		defaultXLAInt(stateDict.DilationD, stateDict.Dilation),
		defaultXLAInt(stateDict.DilationH, stateDict.Dilation),
		defaultXLAInt(stateDict.DilationW, stateDict.Dilation),
		defaultXLAInt(stateDict.Groups, 1), stateDict.Dim0, stateDict.OutH, stateDict.OutW,
	)

	return setXLAOutput(stateDict, output, err)
}

func (convolution *ConvTranspose2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireXLAShape(
		stateDict, "xla.convolution.conv_transpose2d", 4, 1,
	)

	if err != nil {
		return nil, err
	}

	output, err := convolution.convolution.ConvTranspose2d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultXLAInt(stateDict.KernelH, stateDict.KernelSize),
		defaultXLAInt(stateDict.KernelW, stateDict.KernelSize),
		defaultXLAInt(stateDict.StrideH, stateDict.Stride),
		defaultXLAInt(stateDict.StrideW, stateDict.Stride),
		stateDict.PadH, stateDict.PadW,
		defaultXLAInt(stateDict.DilationH, stateDict.Dilation),
		defaultXLAInt(stateDict.DilationW, stateDict.Dilation),
		defaultXLAInt(stateDict.Groups, 1), stateDict.OutH, stateDict.OutW,
	)

	return setXLAOutput(stateDict, output, err)
}

type MaxPool2D struct{ pooling *XLAPooling }
type AvgPool2D struct{ pooling *XLAPooling }
type AdaptiveAvgPool2D struct{ pooling *XLAPooling }
type AdaptiveMaxPool2D struct{ pooling *XLAPooling }

func (pooling *MaxPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.pooling.max_pool2d"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.pooling.max_pool2d: input[0] is required")
	}

	output, err := pooling.pooling.MaxPool2d(
		stateDict.OperationShape(), xlaMaxPoolParams(stateDict), stateDict.Inputs[0],
	)

	return setXLAOutput(stateDict, output, err)
}

func (pooling *AvgPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.pooling.avg_pool2d"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.pooling.avg_pool2d: input[0] is required")
	}

	output, err := pooling.pooling.AvgPool2d(
		stateDict.OperationShape(), xlaAvgPoolParams(stateDict), stateDict.Inputs[0],
	)

	return setXLAOutput(stateDict, output, err)
}

func (pooling *AdaptiveAvgPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.pooling.adaptive_avg_pool2d"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.pooling.adaptive_avg_pool2d: input[0] is required")
	}

	output, err := pooling.pooling.AdaptiveAvgPool2d(
		stateDict.OperationShape(), stateDict.OutH, stateDict.OutW, stateDict.Inputs[0],
	)

	return setXLAOutput(stateDict, output, err)
}

func (pooling *AdaptiveMaxPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.pooling.adaptive_max_pool2d"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.pooling.adaptive_max_pool2d: input[0] is required")
	}

	output, err := pooling.pooling.AdaptiveMaxPool2d(
		stateDict.OperationShape(), stateDict.OutH, stateDict.OutW, stateDict.Inputs[0],
	)

	return setXLAOutput(stateDict, output, err)
}

type Linear struct{ projection *XLAProjection }
type FusedQKV struct{ projection *XLAProjection }

func (linear *Linear) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.projection.linear"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.projection.linear: input[0] is required")
	}

	weight, bias := xlaWeightBias(stateDict)
	output, err := linear.projection.Linear(
		stateDict.OperationShape(), weight, bias, stateDict.Inputs[0],
	)

	return setXLAOutput(stateDict, output, err)
}

func (fusedQKV *FusedQKV) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.projection.fused_qkv"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.projection.fused_qkv: input[0] is required")
	}

	weight, bias := xlaWeightBias(stateDict)
	output, err := fusedQKV.projection.FusedQKV(
		stateDict.OperationShape(), weight, bias, stateDict.Inputs[0],
	)

	return setXLAOutput(stateDict, output, err)
}

type HawkesIntensity struct{ hawkes *XLAHawkes }
type HawkesKernelMatrix struct{ hawkes *XLAHawkes }
type HawkesLogLikelihood struct{ hawkes *XLAHawkes }
type HawkesSimulate struct{ hawkes *XLAHawkes }

func (hawkes *HawkesIntensity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.hawkes.intensity", 5, hawkes.hawkes.Intensity)
}

func (hawkes *HawkesKernelMatrix) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.hawkes.kernel_matrix", 3, hawkes.hawkes.KernelMatrix,
	)
}

func (hawkes *HawkesLogLikelihood) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.hawkes.log_likelihood", 3, hawkes.hawkes.LogLikelihood,
	)
}

func (hawkes *HawkesSimulate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.hawkes.simulate", 4, hawkes.hawkes.Simulate)
}

type VSABind struct{ vsaOps *XLAVSAOps }
type VSABundle struct{ vsaOps *XLAVSAOps }
type VSASimilarity struct{ vsaOps *XLAVSAOps }
type VSAPermute struct{ vsaOps *XLAVSAOps }
type VSAInversePermute struct{ vsaOps *XLAVSAOps }

func (vsaBind *VSABind) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.vsa.bind", 2, vsaBind.vsaOps.Bind)
}

func (vsaBundle *VSABundle) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.vsa.bundle", 1, vsaBundle.vsaOps.Bundle)
}

func (vsaSimilarity *VSASimilarity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.vsa.similarity", 2, vsaSimilarity.vsaOps.Similarity)
}

func (vsaPermute *VSAPermute) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.vsa.permute"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.vsa.permute: input[0] is required")
	}

	output, err := vsaPermute.vsaOps.Permute(
		stateDict.OperationShape(), stateDict.Dim, stateDict.Inputs...,
	)

	return setXLAOutput(stateDict, output, err)
}

func (vsaInversePermute *VSAInversePermute) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("xla.vsa.inverse_permute"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("xla.vsa.inverse_permute: input[0] is required")
	}

	output, err := vsaInversePermute.vsaOps.InversePermute(
		stateDict.OperationShape(), stateDict.Dim, stateDict.Inputs...,
	)

	return setXLAOutput(stateDict, output, err)
}

type BeliefUpdate struct{ activeInference *XLAActiveInferenceOps }
type ExpectedFreeEnergy struct{ activeInference *XLAActiveInferenceOps }
type FreeEnergy struct{ activeInference *XLAActiveInferenceOps }
type PrecisionWeight struct{ activeInference *XLAActiveInferenceOps }

func (activeInference *BeliefUpdate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.active_inference.belief_update", 3,
		activeInference.activeInference.BeliefUpdate,
	)
}

func (activeInference *ExpectedFreeEnergy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.active_inference.expected_free_energy", 1,
		activeInference.activeInference.ExpectedFreeEnergy,
	)
}

func (activeInference *FreeEnergy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.active_inference.free_energy", 2,
		activeInference.activeInference.FreeEnergy,
	)
}

func (activeInference *PrecisionWeight) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.active_inference.precision_weight", 2,
		activeInference.activeInference.PrecisionWeight,
	)
}

type Prediction struct{ predictiveCoding *XLAPredictiveCodingOps }
type PredictionError struct{ predictiveCoding *XLAPredictiveCodingOps }
type UpdateRepresentation struct{ predictiveCoding *XLAPredictiveCodingOps }
type UpdateWeights struct{ predictiveCoding *XLAPredictiveCodingOps }

func (prediction *Prediction) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.predictive_coding.prediction", 2,
		prediction.predictiveCoding.Prediction,
	)
}

func (predictionError *PredictionError) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.predictive_coding.prediction_error", 2,
		predictionError.predictiveCoding.PredictionError,
	)
}

func (updateRepresentation *UpdateRepresentation) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.predictive_coding.update_representation", 5,
		updateRepresentation.predictiveCoding.UpdateRepresentation,
	)
}

func (updateWeights *UpdateWeights) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.predictive_coding.update_weights", 4,
		updateWeights.predictiveCoding.UpdateWeights,
	)
}

type FlowActive struct{ markovBlanket *XLAMarkovBlanket }
type FlowInternal struct{ markovBlanket *XLAMarkovBlanket }
type MutualInformation struct{ markovBlanket *XLAMarkovBlanket }
type Partition struct{ markovBlanket *XLAMarkovBlanket }

func (markovBlanket *FlowActive) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.markov_blanket.flow_active", 3,
		markovBlanket.markovBlanket.FlowActive,
	)
}

func (markovBlanket *FlowInternal) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.markov_blanket.flow_internal", 3,
		markovBlanket.markovBlanket.FlowInternal,
	)
}

func (markovBlanket *MutualInformation) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.markov_blanket.mutual_information", 2,
		markovBlanket.markovBlanket.MutualInformation,
	)
}

func (markovBlanket *Partition) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.markov_blanket.partition", 2,
		markovBlanket.markovBlanket.Partition,
	)
}

type BackdoorAdjustment struct{ causal *XLACausalOps }
type CATE struct{ causal *XLACausalOps }
type Counterfactual struct{ causal *XLACausalOps }
type DAGMarkovFactorization struct{ causal *XLACausalOps }
type DoCalculus struct{ causal *XLACausalOps }
type FrontdoorAdjustment struct{ causal *XLACausalOps }
type IVEstimate struct{ causal *XLACausalOps }

func (causal *BackdoorAdjustment) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.causal.backdoor_adjustment", 3, causal.causal.BackdoorAdjustment,
	)
}

func (causal *CATE) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.causal.cate", 3, causal.causal.CATE)
}

func (causal *Counterfactual) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.causal.counterfactual", 4, causal.causal.Counterfactual,
	)
}

func (causal *DAGMarkovFactorization) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.causal.dag_markov_factorization", 2,
		causal.causal.DAGMarkovFactorization,
	)
}

func (causal *DoCalculus) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.causal.do_calculus", 3, causal.causal.DoCalculus)
}

func (causal *FrontdoorAdjustment) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(
		stateDict, "xla.causal.frontdoor_adjustment", 3,
		causal.causal.FrontdoorAdjustment,
	)
}

func (causal *IVEstimate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return xlaShapeForward(stateDict, "xla.causal.iv_estimate", 3, causal.causal.IVEstimate)
}

type MSELoss struct{ trainingOps *XLATrainingOps }
type CrossEntropyLoss struct{ trainingOps *XLATrainingOps }
type MSEGrad struct{ trainingOps *XLATrainingOps }
type CrossEntropyGrad struct{ trainingOps *XLATrainingOps }
type Accuracy struct{ trainingOps *XLATrainingOps }
type Perplexity struct{ trainingOps *XLATrainingOps }
type F1 struct{ trainingOps *XLATrainingOps }

func (mseLoss *MSELoss) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := xlaTrainInputs(stateDict, "xla.train.mse_loss"); err != nil {
		return nil, err
	}

	output, err := mseLoss.trainingOps.MSELoss(stateDict.Inputs[0], stateDict.Inputs[1])
	return setXLAOutput(stateDict, output, err)
}

func (crossEntropyLoss *CrossEntropyLoss) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	if err := xlaTrainInputs(stateDict, "xla.train.cross_entropy_loss"); err != nil {
		return nil, err
	}

	output, err := crossEntropyLoss.trainingOps.CrossEntropyLoss(
		stateDict.Inputs[0], stateDict.Inputs[1],
	)

	return setXLAOutput(stateDict, output, err)
}

func (mseGrad *MSEGrad) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := xlaTrainInputs(stateDict, "xla.train.mse_grad"); err != nil {
		return nil, err
	}

	output, err := mseGrad.trainingOps.MSEGrad(stateDict.Inputs[0], stateDict.Inputs[1])
	return setXLAOutput(stateDict, output, err)
}

func (crossEntropyGrad *CrossEntropyGrad) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	if err := xlaTrainInputs(stateDict, "xla.train.cross_entropy_grad"); err != nil {
		return nil, err
	}

	output, err := crossEntropyGrad.trainingOps.CrossEntropyGrad(
		stateDict.Inputs[0], stateDict.Inputs[1],
	)

	return setXLAOutput(stateDict, output, err)
}

func (accuracy *Accuracy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := xlaTrainInputs(stateDict, "xla.bench.accuracy"); err != nil {
		return nil, err
	}

	output, err := accuracy.trainingOps.Accuracy(stateDict.Inputs[0], stateDict.Inputs[1])

	if err != nil {
		return nil, err
	}

	stateDict.RecordAccuracy(output[0] == 1)

	return stateDict, nil
}

func (perplexity *Perplexity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := xlaTrainInputs(stateDict, "xla.bench.perplexity"); err != nil {
		return nil, err
	}

	output, err := perplexity.trainingOps.CrossEntropyLoss(
		stateDict.Inputs[0], stateDict.Inputs[1],
	)

	if err != nil {
		return nil, err
	}

	stateDict.RecordPerplexity(output[0])

	return stateDict, nil
}

func (f1 *F1) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := xlaTrainInputs(stateDict, "xla.bench.f1"); err != nil {
		return nil, err
	}

	counts, err := f1.trainingOps.F1Counts(stateDict.Inputs[0], stateDict.Inputs[1])

	if err != nil {
		return nil, err
	}

	stateDict.RecordF1(counts[0], counts[1], counts[2])

	return stateDict, nil
}

func xlaUnaryForward(
	stateDict *state.Dict,
	name string,
	forward func([]float64) ([]float64, error),
) (*state.Dict, error) {
	if err := stateDict.RequireOperation(name); err != nil {
		return nil, err
	}

	if stateDict.Inputs == nil || len(stateDict.Inputs) == 0 {
		return nil, fmt.Errorf("missing input for operation %s", name)
	}

	output, err := forward(stateDict.Inputs[0])
	return setXLAOutput(stateDict, output, err)
}

func xlaShapeForward(
	stateDict *state.Dict,
	name string,
	inputCount int,
	forward func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs(name, inputCount); err != nil {
		return nil, err
	}

	output, err := forward(stateDict.OperationShape(), stateDict.Inputs...)
	return setXLAOutput(stateDict, output, err)
}

func xlaTrainInputs(stateDict *state.Dict, name string) error {
	if err := stateDict.RequireOperationInputs(name, 2); err != nil {
		return err
	}

	if len(stateDict.Inputs[0]) == 0 || len(stateDict.Inputs[0]) != len(stateDict.Inputs[1]) {
		return fmt.Errorf("%s: input lengths must match and be non-zero", name)
	}

	return nil
}

func setXLAOutput(
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

func requireXLAShape(
	stateDict *state.Dict,
	name string,
	rank int,
	inputCount int,
) ([]int, error) {
	if err := stateDict.RequireOperationInputs(name, inputCount); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < rank {
		return nil, fmt.Errorf("%s: shape rank %d, need >= %d", name, len(shape), rank)
	}

	return shape, nil
}

func xlaWeightBias(stateDict *state.Dict) ([]float64, []float64) {
	weight := stateDict.Weight

	if len(weight) == 0 && len(stateDict.Inputs) > 1 {
		weight = stateDict.Inputs[1]
	}

	bias := stateDict.Bias

	if len(bias) == 0 && len(stateDict.Inputs) > 2 {
		bias = stateDict.Inputs[2]
	}

	return weight, bias
}

func xlaMaxPoolParams(stateDict *state.Dict) XLAMaxPool2dParams {
	return XLAMaxPool2dParams{
		KernelH:   defaultXLAInt(stateDict.KernelH, stateDict.KernelSize),
		KernelW:   defaultXLAInt(stateDict.KernelW, stateDict.KernelSize),
		StrideH:   defaultXLAInt(stateDict.StrideH, stateDict.Stride),
		StrideW:   defaultXLAInt(stateDict.StrideW, stateDict.Stride),
		PadH:      stateDict.PadH,
		PadW:      stateDict.PadW,
		DilationH: defaultXLAInt(stateDict.DilationH, stateDict.Dilation),
		DilationW: defaultXLAInt(stateDict.DilationW, stateDict.Dilation),
		CeilMode:  stateDict.Ceil,
	}
}

func xlaAvgPoolParams(stateDict *state.Dict) XLAAvgPool2dParams {
	return XLAAvgPool2dParams{
		KernelH:         defaultXLAInt(stateDict.KernelH, stateDict.KernelSize),
		KernelW:         defaultXLAInt(stateDict.KernelW, stateDict.KernelSize),
		StrideH:         defaultXLAInt(stateDict.StrideH, stateDict.Stride),
		StrideW:         defaultXLAInt(stateDict.StrideW, stateDict.Stride),
		PadH:            stateDict.PadH,
		PadW:            stateDict.PadW,
		DilationH:       defaultXLAInt(stateDict.DilationH, stateDict.Dilation),
		DilationW:       defaultXLAInt(stateDict.DilationW, stateDict.Dilation),
		CeilMode:        stateDict.Ceil,
		CountIncludePad: stateDict.CountPad,
		DivisorOverride: stateDict.Divisor,
	}
}

func xlaStateConfig(config *state.Dict) *state.Dict {
	if config == nil {
		return state.NewDict()
	}

	return config
}

func xlaPlatform(config *state.Dict) string {
	config = xlaStateConfig(config)

	if config.Mode != "" {
		return config.Mode
	}

	return "cpu"
}

func newXLAActivation(config *state.Dict) (*XLAActivation, error) {
	return New(xlaPlatform(config))
}

func newXLAShapeOps(config *state.Dict) (*XLAShapeOps, error) {
	if _, err := New(xlaPlatform(config)); err != nil {
		return nil, err
	}

	return NewShapeOps(), nil
}

func defaultXLAInt(value, fallback int) int {
	if value == 0 {
		return fallback
	}

	return value
}

func defaultXLAFloat(value, fallback float64) float64 {
	if value == 0 {
		return fallback
	}

	return value
}
