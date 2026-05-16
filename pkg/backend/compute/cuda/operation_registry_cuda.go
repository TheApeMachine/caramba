//go:build linux && cgo && cuda

package cuda

import (
	"fmt"
	stdmath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func (registry OperationRegistry) ReLU(*state.Dict) (state.Operation, error) {
	return &ReLU{activation: New()}, nil
}

func (registry OperationRegistry) LeakyReLU(*state.Dict) (state.Operation, error) {
	return &LeakyReLU{activation: New()}, nil
}

func (registry OperationRegistry) GELU(*state.Dict) (state.Operation, error) {
	return &GELU{activation: New()}, nil
}

func (registry OperationRegistry) Tanh(*state.Dict) (state.Operation, error) {
	return &Tanh{activation: New()}, nil
}

func (registry OperationRegistry) Sigmoid(*state.Dict) (state.Operation, error) {
	return &Sigmoid{activation: New()}, nil
}

func (registry OperationRegistry) SwiGLU(*state.Dict) (state.Operation, error) {
	return &SwiGLU{activation: New()}, nil
}

func (registry OperationRegistry) Swish(*state.Dict) (state.Operation, error) {
	return &Swish{activation: New()}, nil
}

func (registry OperationRegistry) SELU(*state.Dict) (state.Operation, error) {
	return &SELU{activation: New()}, nil
}

func (registry OperationRegistry) SDPA(*state.Dict) (state.Operation, error) {
	return &SDPA{attention: NewAttention()}, nil
}

func (registry OperationRegistry) MQA(*state.Dict) (state.Operation, error) {
	return &MQA{attention: NewAttention()}, nil
}

func (registry OperationRegistry) GQA(*state.Dict) (state.Operation, error) {
	return &GQA{attention: NewAttention()}, nil
}

func (registry OperationRegistry) SlidingWindowAttention(*state.Dict) (state.Operation, error) {
	return &SlidingWindowAttention{attention: NewAttention()}, nil
}

func (registry OperationRegistry) ApplyMask(*state.Dict) (state.Operation, error) {
	return &ApplyMask{masking: NewMasking()}, nil
}

func (registry OperationRegistry) CausalMask(*state.Dict) (state.Operation, error) {
	return &CausalMask{masking: NewMasking()}, nil
}

func (registry OperationRegistry) Add(*state.Dict) (state.Operation, error) {
	return &Add{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Mul(*state.Dict) (state.Operation, error) {
	return &Mul{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Matmul(*state.Dict) (state.Operation, error) {
	return &Matmul{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Exp(*state.Dict) (state.Operation, error) {
	return &Exp{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Log(*state.Dict) (state.Operation, error) {
	return &Log{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) LogSumExp(*state.Dict) (state.Operation, error) {
	return &LogSumExp{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Softmax(*state.Dict) (state.Operation, error) {
	return &Softmax{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Outer(*state.Dict) (state.Operation, error) {
	return &Outer{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Sign(*state.Dict) (state.Operation, error) {
	return &Sign{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) InvSqrtDimScale(*state.Dict) (state.Operation, error) {
	return &InvSqrtDimScale{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Dropout(*state.Dict) (state.Operation, error) {
	return &Dropout{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) RMSNorm(*state.Dict) (state.Operation, error) {
	return &RMSNorm{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) LayerNorm(*state.Dict) (state.Operation, error) {
	return &LayerNorm{mathOps: NewMathOps()}, nil
}

func (registry OperationRegistry) Reshape(*state.Dict) (state.Operation, error) {
	return &Reshape{shapeOps: NewShapeOps()}, nil
}

func (registry OperationRegistry) Transpose(*state.Dict) (state.Operation, error) {
	return &Transpose{shapeOps: NewShapeOps()}, nil
}

func (registry OperationRegistry) Concat(*state.Dict) (state.Operation, error) {
	return &Concat{shapeOps: NewShapeOps()}, nil
}

func (registry OperationRegistry) Split(*state.Dict) (state.Operation, error) {
	return &Split{shapeOps: NewShapeOps()}, nil
}

func (registry OperationRegistry) ViewAsHeads(*state.Dict) (state.Operation, error) {
	return &ViewAsHeads{shapeOps: NewShapeOps()}, nil
}

func (registry OperationRegistry) MergeHeads(*state.Dict) (state.Operation, error) {
	return &MergeHeads{shapeOps: NewShapeOps()}, nil
}

func (registry OperationRegistry) LastToken(*state.Dict) (state.Operation, error) {
	return &LastToken{shapeOps: NewShapeOps()}, nil
}

func (registry OperationRegistry) RoPE(*state.Dict) (state.Operation, error) {
	return &RoPE{positional: NewPositional()}, nil
}

func (registry OperationRegistry) ALiBi(*state.Dict) (state.Operation, error) {
	return &ALiBi{positional: NewPositional()}, nil
}

func (registry OperationRegistry) TokenEmbedding(config *state.Dict) (state.Operation, error) {
	config = cudaStateConfig(config)
	return &TokenEmbedding{
		embedding: NewCUDAEmbedding(config.VocabSize, config.DModel),
	}, nil
}

func (registry OperationRegistry) Conv1D(*state.Dict) (state.Operation, error) {
	return &Conv1D{convolution: NewCUDAConvolution()}, nil
}

func (registry OperationRegistry) Conv2D(*state.Dict) (state.Operation, error) {
	return &Conv2D{convolution: NewCUDAConvolution()}, nil
}

func (registry OperationRegistry) Conv3D(*state.Dict) (state.Operation, error) {
	return &Conv3D{convolution: NewCUDAConvolution()}, nil
}

func (registry OperationRegistry) ConvTranspose2D(*state.Dict) (state.Operation, error) {
	return &ConvTranspose2D{convolution: NewCUDAConvolution()}, nil
}

func (registry OperationRegistry) MaxPool2D(*state.Dict) (state.Operation, error) {
	return &MaxPool2D{pooling: NewCUDAPooling()}, nil
}

func (registry OperationRegistry) AvgPool2D(*state.Dict) (state.Operation, error) {
	return &AvgPool2D{pooling: NewCUDAPooling()}, nil
}

func (registry OperationRegistry) AdaptiveAvgPool2D(*state.Dict) (state.Operation, error) {
	return &AdaptiveAvgPool2D{pooling: NewCUDAPooling()}, nil
}

func (registry OperationRegistry) AdaptiveMaxPool2D(*state.Dict) (state.Operation, error) {
	return &AdaptiveMaxPool2D{pooling: NewCUDAPooling()}, nil
}

func (registry OperationRegistry) Linear(*state.Dict) (state.Operation, error) {
	return &Linear{projection: NewCUDAProjection()}, nil
}

func (registry OperationRegistry) FusedQKV(*state.Dict) (state.Operation, error) {
	return &FusedQKV{projection: NewCUDAProjection()}, nil
}

func (registry OperationRegistry) HawkesIntensity(*state.Dict) (state.Operation, error) {
	return &HawkesIntensity{hawkes: NewHawkes()}, nil
}

func (registry OperationRegistry) HawkesKernelMatrix(*state.Dict) (state.Operation, error) {
	return &HawkesKernelMatrix{hawkes: NewHawkes()}, nil
}

func (registry OperationRegistry) HawkesLogLikelihood(*state.Dict) (state.Operation, error) {
	return &HawkesLogLikelihood{hawkes: NewHawkes()}, nil
}

func (registry OperationRegistry) HawkesSimulate(*state.Dict) (state.Operation, error) {
	return &HawkesSimulate{hawkes: NewHawkes()}, nil
}

func (registry OperationRegistry) VSABind(*state.Dict) (state.Operation, error) {
	return &VSABind{vsaOps: NewVSAOps()}, nil
}

func (registry OperationRegistry) VSABundle(*state.Dict) (state.Operation, error) {
	return &VSABundle{vsaOps: NewVSAOps()}, nil
}

func (registry OperationRegistry) VSASimilarity(*state.Dict) (state.Operation, error) {
	return &VSASimilarity{vsaOps: NewVSAOps()}, nil
}

func (registry OperationRegistry) VSAPermute(*state.Dict) (state.Operation, error) {
	return &VSAPermute{vsaOps: NewVSAOps()}, nil
}

func (registry OperationRegistry) VSAInversePermute(*state.Dict) (state.Operation, error) {
	return &VSAInversePermute{vsaOps: NewVSAOps()}, nil
}

func (registry OperationRegistry) BeliefUpdate(*state.Dict) (state.Operation, error) {
	return &BeliefUpdate{activeInference: NewActiveInferenceOps()}, nil
}

func (registry OperationRegistry) ExpectedFreeEnergy(*state.Dict) (state.Operation, error) {
	return &ExpectedFreeEnergy{activeInference: NewActiveInferenceOps()}, nil
}

func (registry OperationRegistry) FreeEnergy(*state.Dict) (state.Operation, error) {
	return &FreeEnergy{activeInference: NewActiveInferenceOps()}, nil
}

func (registry OperationRegistry) PrecisionWeight(*state.Dict) (state.Operation, error) {
	return &PrecisionWeight{activeInference: NewActiveInferenceOps()}, nil
}

func (registry OperationRegistry) Prediction(*state.Dict) (state.Operation, error) {
	return &Prediction{predictiveCoding: NewPredictiveCodingOps()}, nil
}

func (registry OperationRegistry) PredictionError(*state.Dict) (state.Operation, error) {
	return &PredictionError{predictiveCoding: NewPredictiveCodingOps()}, nil
}

func (registry OperationRegistry) UpdateRepresentation(*state.Dict) (state.Operation, error) {
	return &UpdateRepresentation{predictiveCoding: NewPredictiveCodingOps()}, nil
}

func (registry OperationRegistry) UpdateWeights(*state.Dict) (state.Operation, error) {
	return &UpdateWeights{predictiveCoding: NewPredictiveCodingOps()}, nil
}

func (registry OperationRegistry) FlowActive(*state.Dict) (state.Operation, error) {
	return &FlowActive{markovBlanket: NewMarkovBlanket()}, nil
}

func (registry OperationRegistry) FlowInternal(*state.Dict) (state.Operation, error) {
	return &FlowInternal{markovBlanket: NewMarkovBlanket()}, nil
}

func (registry OperationRegistry) MutualInformation(*state.Dict) (state.Operation, error) {
	return &MutualInformation{markovBlanket: NewMarkovBlanket()}, nil
}

func (registry OperationRegistry) Partition(*state.Dict) (state.Operation, error) {
	return &Partition{markovBlanket: NewMarkovBlanket()}, nil
}

func (registry OperationRegistry) BackdoorAdjustment(*state.Dict) (state.Operation, error) {
	return &BackdoorAdjustment{causal: NewCausalOps()}, nil
}

func (registry OperationRegistry) CATE(*state.Dict) (state.Operation, error) {
	return &CATE{causal: NewCausalOps()}, nil
}

func (registry OperationRegistry) Counterfactual(*state.Dict) (state.Operation, error) {
	return &Counterfactual{causal: NewCausalOps()}, nil
}

func (registry OperationRegistry) DAGMarkovFactorization(*state.Dict) (state.Operation, error) {
	return &DAGMarkovFactorization{causal: NewCausalOps()}, nil
}

func (registry OperationRegistry) DoCalculus(*state.Dict) (state.Operation, error) {
	return &DoCalculus{causal: NewCausalOps()}, nil
}

func (registry OperationRegistry) FrontdoorAdjustment(*state.Dict) (state.Operation, error) {
	return &FrontdoorAdjustment{causal: NewCausalOps()}, nil
}

func (registry OperationRegistry) IVEstimate(*state.Dict) (state.Operation, error) {
	return &IVEstimate{causal: NewCausalOps()}, nil
}

func (registry OperationRegistry) MSELoss(*state.Dict) (state.Operation, error) {
	return &MSELoss{trainingOps: NewTrainingOps()}, nil
}

func (registry OperationRegistry) CrossEntropyLoss(*state.Dict) (state.Operation, error) {
	return &CrossEntropyLoss{trainingOps: NewTrainingOps()}, nil
}

func (registry OperationRegistry) MSEGrad(*state.Dict) (state.Operation, error) {
	return &MSEGrad{trainingOps: NewTrainingOps()}, nil
}

func (registry OperationRegistry) CrossEntropyGrad(*state.Dict) (state.Operation, error) {
	return &CrossEntropyGrad{trainingOps: NewTrainingOps()}, nil
}

func (registry OperationRegistry) Accuracy(*state.Dict) (state.Operation, error) {
	return &Accuracy{trainingOps: NewTrainingOps()}, nil
}

func (registry OperationRegistry) Perplexity(*state.Dict) (state.Operation, error) {
	return &Perplexity{trainingOps: NewTrainingOps()}, nil
}

func (registry OperationRegistry) F1(*state.Dict) (state.Operation, error) {
	return &F1{trainingOps: NewTrainingOps()}, nil
}

func (registry OperationRegistry) Load(config *state.Dict) (state.Operation, error) {
	config = cudaStateConfig(config)
	return NewCUDAModelOps().NewLoader(config.Source, config.File, config.Cache), nil
}

func (registry OperationRegistry) Surgery(config *state.Dict) (state.Operation, error) {
	config = cudaStateConfig(config)
	return NewCUDAModelOps().NewSurgery(
		config.Source, config.Op, config.At, config.After, config.Name, config.Layer,
	), nil
}

func (registry OperationRegistry) Graft(config *state.Dict) (state.Operation, error) {
	config = cudaStateConfig(config)
	return NewCUDAModelOps().NewGraft(config.Source, config.At, config.Mode), nil
}

func (registry OperationRegistry) LoRA(config *state.Dict) (state.Operation, error) {
	config = cudaStateConfig(config)
	return NewCUDAModelOps().NewLoRA(
		config.Source, config.Preset, config.Targets, config.Rank, config.Alpha,
	), nil
}

func (registry OperationRegistry) Adapter(config *state.Dict) (state.Operation, error) {
	config = cudaStateConfig(config)
	return NewCUDAModelOps().NewAdapter(config.Source, config.At, config.Reduction), nil
}

func (registry OperationRegistry) Freeze(config *state.Dict) (state.Operation, error) {
	config = cudaStateConfig(config)
	return NewCUDAModelOps().NewFreeze(
		config.Source, config.Pattern, config.Except, config.Frozen,
	), nil
}

type ReLU struct{ activation *CUDAActivation }
type LeakyReLU struct{ activation *CUDAActivation }
type GELU struct{ activation *CUDAActivation }
type Tanh struct{ activation *CUDAActivation }
type Sigmoid struct{ activation *CUDAActivation }
type Swish struct{ activation *CUDAActivation }
type SwiGLU struct{ activation *CUDAActivation }
type SELU struct{ activation *CUDAActivation }

func (relu *ReLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaUnaryForward(stateDict, "cuda.activation.relu", relu.activation.ReLU)
}

func (leakyReLU *LeakyReLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaUnaryForward(
		stateDict,
		"cuda.activation.leaky_relu",
		func(input []float64) ([]float64, error) {
			return leakyReLU.activation.LeakyReLU(input, stateDict.Alpha)
		},
	)
}

func (gelu *GELU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaUnaryForward(stateDict, "cuda.activation.gelu", gelu.activation.GELU)
}

func (tanh *Tanh) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaUnaryForward(stateDict, "cuda.activation.tanh", tanh.activation.Tanh)
}

func (sigmoid *Sigmoid) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaUnaryForward(stateDict, "cuda.activation.sigmoid", sigmoid.activation.Sigmoid)
}

func (swish *Swish) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaUnaryForward(stateDict, "cuda.activation.swish", swish.activation.Swish)
}

func (swiglu *SwiGLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaUnaryForward(stateDict, "cuda.activation.swiglu", swiglu.activation.SwiGLU)
}

func (selu *SELU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaUnaryForward(stateDict, "cuda.activation.selu", selu.activation.SELU)
}

type SDPA struct{ attention *CUDAAttention }
type MQA struct{ attention *CUDAAttention }
type GQA struct{ attention *CUDAAttention }
type SlidingWindowAttention struct{ attention *CUDAAttention }

func (sdpa *SDPA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(stateDict, "cuda.attention.sdpa", 4, 3)

	if err != nil {
		return nil, err
	}

	output, err := sdpa.attention.SDPA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (mqa *MQA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(stateDict, "cuda.attention.mqa", 4, 3)

	if err != nil {
		return nil, err
	}

	output, err := mqa.attention.MQA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (gqa *GQA) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(stateDict, "cuda.attention.gqa", 5, 3)

	if err != nil {
		return nil, err
	}

	output, err := gqa.attention.GQA(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3], shape[4],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (attention *SlidingWindowAttention) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(
		stateDict, "cuda.attention.sliding_window_attention", 4, 3,
	)

	if err != nil {
		return nil, err
	}

	output, err := attention.attention.SlidingWindow(
		stateDict.Inputs[0], stateDict.Inputs[1], stateDict.Inputs[2],
		shape[0], shape[1], shape[2], shape[3], stateDict.Window,
	)

	return setCUDAOutput(stateDict, output, err)
}

type ApplyMask struct{ masking *CUDAMasking }
type CausalMask struct{ masking *CUDAMasking }

func (applyMask *ApplyMask) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("cuda.masking.apply_mask", 2); err != nil {
		return nil, err
	}

	output, err := applyMask.masking.ApplyMask(stateDict.Inputs[0], stateDict.Inputs[1])
	return setCUDAOutput(stateDict, output, err)
}

func (causalMask *CausalMask) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.masking.causal_mask"); err != nil {
		return nil, err
	}

	output, err := causalMask.masking.CausalMask(stateDict.OperationLastDim())
	return setCUDAOutput(stateDict, output, err)
}

type Add struct{ mathOps *CUDAMathOps }
type Mul struct{ mathOps *CUDAMathOps }
type Matmul struct{ mathOps *CUDAMathOps }
type Exp struct{ mathOps *CUDAMathOps }
type Log struct{ mathOps *CUDAMathOps }
type LogSumExp struct{ mathOps *CUDAMathOps }
type Softmax struct{ mathOps *CUDAMathOps }
type Outer struct{ mathOps *CUDAMathOps }
type Sign struct{ mathOps *CUDAMathOps }
type InvSqrtDimScale struct{ mathOps *CUDAMathOps }
type Dropout struct{ mathOps *CUDAMathOps }
type RMSNorm struct{ mathOps *CUDAMathOps }
type LayerNorm struct{ mathOps *CUDAMathOps }

func (add *Add) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.math.add", 2, add.mathOps.Add)
}

func (mul *Mul) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.math.mul", 2, mul.mathOps.Mul)
}

func (matmul *Matmul) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.math.matmul", 2, matmul.mathOps.Matmul)
}

func (exp *Exp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.math.exp", 1, exp.mathOps.Exp)
}

func (log *Log) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.math.log", 1, log.mathOps.Log)
}

func (logSumExp *LogSumExp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.math.logsumexp"); err != nil {
		return nil, err
	}

	dimSize := stateDict.OperationLastDim()

	if dimSize <= 0 {
		return nil, fmt.Errorf("cuda.math.logsumexp: last dimension must be positive")
	}

	if len(stateDict.Inputs[0])%dimSize != 0 {
		return nil, fmt.Errorf("cuda.math.logsumexp: input length must divide last dimension")
	}

	output, err := logSumExp.mathOps.LogSumExp(stateDict.OperationShape(), stateDict.Inputs...)

	return setCUDAOutput(stateDict, output, err)
}

func (softmax *Softmax) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.math.softmax", 1, softmax.mathOps.Softmax)
}

func (outer *Outer) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.math.outer", 2, outer.mathOps.Outer)
}

func (sign *Sign) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.math.sign", 1, sign.mathOps.Sign)
}

func (scale *InvSqrtDimScale) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.math.inv_sqrt_dim_scale", 1, scale.mathOps.InvSqrtDimScale,
	)
}

func (dropout *Dropout) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.math.dropout"); err != nil {
		return nil, err
	}

	output, err := dropout.mathOps.Dropout(
		stateDict.P, stateDict.Training, stateDict.Step, stateDict.Inputs[0],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (rmsNorm *RMSNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.math.rmsnorm"); err != nil {
		return nil, err
	}

	output, err := rmsNorm.mathOps.RMSNorm(
		stateDict.OperationShape(), stateDict.Eps, stateDict.Weight, stateDict.Inputs...,
	)

	return setCUDAOutput(stateDict, output, err)
}

func (layerNorm *LayerNorm) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.math.layernorm"); err != nil {
		return nil, err
	}

	output, err := layerNorm.mathOps.LayerNorm(
		stateDict.OperationShape(), stateDict.Eps, stateDict.Weight, stateDict.Bias,
		stateDict.Inputs...,
	)

	return setCUDAOutput(stateDict, output, err)
}

type Reshape struct{ shapeOps *CUDAShapeOps }
type Transpose struct{ shapeOps *CUDAShapeOps }
type Concat struct{ shapeOps *CUDAShapeOps }
type Split struct{ shapeOps *CUDAShapeOps }
type ViewAsHeads struct{ shapeOps *CUDAShapeOps }
type MergeHeads struct{ shapeOps *CUDAShapeOps }
type LastToken struct{ shapeOps *CUDAShapeOps }

func (reshape *Reshape) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.shape.reshape", 1, func(
		_ []int, data ...[]float64,
	) ([]float64, error) {
		return reshape.shapeOps.Copy(data[0])
	})
}

func (transpose *Transpose) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.shape.transpose"); err != nil {
		return nil, err
	}

	output, err := transpose.shapeOps.Transpose(
		stateDict.OperationShape(), stateDict.Dim0, stateDict.Dim1, stateDict.Inputs[0],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (concat *Concat) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("cuda.shape.concat", 2); err != nil {
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
	if err := stateDict.RequireOperation("cuda.shape.split"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()
	rank := len(shape)

	if stateDict.Dim < 0 || stateDict.Dim >= rank {
		return nil, fmt.Errorf("cuda.shape.split: dim %d out of range rank %d", stateDict.Dim, rank)
	}

	if stateDict.SplitSize <= 0 {
		return nil, fmt.Errorf("cuda.shape.split: split size must be positive")
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
		return nil, fmt.Errorf("cuda.shape.split: dim size is not divisible by split size")
	}

	output, err := split.shapeOps.Split(
		stateDict.Inputs[0], outer, dimSize, stateDict.SplitSize, inner,
	)

	return setCUDAOutput(stateDict, output, err)
}

func (viewAsHeads *ViewAsHeads) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(stateDict, "cuda.shape.view_as_heads", 3, 1)

	if err != nil {
		return nil, err
	}

	if stateDict.NumHeads <= 0 {
		return nil, fmt.Errorf("cuda.shape.view_as_heads: NumHeads must be > 0")
	}

	if shape[2]%stateDict.NumHeads != 0 {
		return nil, fmt.Errorf(
			"cuda.shape.view_as_heads: model dimension %d not divisible by NumHeads %d",
			shape[2], stateDict.NumHeads,
		)
	}

	output, err := viewAsHeads.shapeOps.ViewAsHeads(
		stateDict.Inputs[0], shape[0], shape[1], stateDict.NumHeads,
		shape[2]/stateDict.NumHeads,
	)

	return setCUDAOutput(stateDict, output, err)
}

func (mergeHeads *MergeHeads) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(stateDict, "cuda.shape.merge_heads", 4, 1)

	if err != nil {
		return nil, err
	}

	output, err := mergeHeads.shapeOps.MergeHeads(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (lastToken *LastToken) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.shape.last_token"); err != nil {
		return nil, err
	}

	outer, sequenceLength, featureLength, err := cudaLastTokenShapeParts(
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

	return setCUDAOutput(stateDict, output, err)
}

type RoPE struct{ positional *CUDAPositionalOps }
type ALiBi struct{ positional *CUDAPositionalOps }

func (rope *RoPE) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.positional.rope"); err != nil {
		return nil, err
	}

	output, err := rope.positional.RoPEForward(
		defaultCUDAFloat(stateDict.Base, 10000), stateDict.OperationShape(), stateDict.Inputs...,
	)

	return setCUDAOutput(stateDict, output, err)
}

func (alibi *ALiBi) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.positional.alibi"); err != nil {
		return nil, err
	}

	output, err := alibi.positional.ALiBiForward(stateDict.OperationShape(), stateDict.Causal)
	return setCUDAOutput(stateDict, output, err)
}

type TokenEmbedding struct{ embedding *CUDAEmbedding }

func (tokenEmbedding *TokenEmbedding) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("cuda.embedding.token_embedding", 2); err != nil {
		return nil, err
	}

	output, err := tokenEmbedding.embedding.Forward(stateDict.OperationShape(), stateDict.Inputs...)
	return setCUDAOutput(stateDict, output, err)
}

type Conv1D struct{ convolution *CUDAConvolution }
type Conv2D struct{ convolution *CUDAConvolution }
type Conv3D struct{ convolution *CUDAConvolution }
type ConvTranspose2D struct{ convolution *CUDAConvolution }

func (convolution *Conv1D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(stateDict, "cuda.convolution.conv1d", 3, 1)

	if err != nil {
		return nil, err
	}

	output, err := convolution.convolution.Conv1d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], stateDict.Weight,
		stateDict.Bias, stateDict.OutChannels, stateDict.KernelSize,
		defaultCUDAInt(stateDict.Stride, 1), stateDict.Padding,
		defaultCUDAInt(stateDict.Dilation, 1), defaultCUDAInt(stateDict.Groups, 1),
		stateDict.OutW,
	)

	return setCUDAOutput(stateDict, output, err)
}

func (convolution *Conv2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(stateDict, "cuda.convolution.conv2d", 4, 1)

	if err != nil {
		return nil, err
	}

	output, err := convolution.convolution.Conv2d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultCUDAInt(stateDict.KernelH, stateDict.KernelSize),
		defaultCUDAInt(stateDict.KernelW, stateDict.KernelSize),
		defaultCUDAInt(stateDict.StrideH, stateDict.Stride),
		defaultCUDAInt(stateDict.StrideW, stateDict.Stride),
		stateDict.PadH, stateDict.PadW,
		defaultCUDAInt(stateDict.DilationH, stateDict.Dilation),
		defaultCUDAInt(stateDict.DilationW, stateDict.Dilation),
		defaultCUDAInt(stateDict.Groups, 1), stateDict.OutH, stateDict.OutW,
	)

	return setCUDAOutput(stateDict, output, err)
}

func (convolution *Conv3D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(stateDict, "cuda.convolution.conv3d", 5, 1)

	if err != nil {
		return nil, err
	}

	output, err := convolution.convolution.Conv3d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3], shape[4],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultCUDAInt(stateDict.KernelD, stateDict.KernelSize),
		defaultCUDAInt(stateDict.KernelH, stateDict.KernelSize),
		defaultCUDAInt(stateDict.KernelW, stateDict.KernelSize),
		defaultCUDAInt(stateDict.StrideD, stateDict.Stride),
		defaultCUDAInt(stateDict.StrideH, stateDict.Stride),
		defaultCUDAInt(stateDict.StrideW, stateDict.Stride),
		stateDict.PadD, stateDict.PadH, stateDict.PadW,
		defaultCUDAInt(stateDict.DilationD, stateDict.Dilation),
		defaultCUDAInt(stateDict.DilationH, stateDict.Dilation),
		defaultCUDAInt(stateDict.DilationW, stateDict.Dilation),
		defaultCUDAInt(stateDict.Groups, 1), stateDict.Dim0, stateDict.OutH, stateDict.OutW,
	)

	return setCUDAOutput(stateDict, output, err)
}

func (convolution *ConvTranspose2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape, err := requireCUDAShape(
		stateDict, "cuda.convolution.conv_transpose2d", 4, 1,
	)

	if err != nil {
		return nil, err
	}

	output, err := convolution.convolution.ConvTranspose2d(
		stateDict.Inputs[0], shape[0], shape[1], shape[2], shape[3],
		stateDict.Weight, stateDict.Bias, stateDict.OutChannels,
		defaultCUDAInt(stateDict.KernelH, stateDict.KernelSize),
		defaultCUDAInt(stateDict.KernelW, stateDict.KernelSize),
		defaultCUDAInt(stateDict.StrideH, stateDict.Stride),
		defaultCUDAInt(stateDict.StrideW, stateDict.Stride),
		stateDict.PadH, stateDict.PadW,
		defaultCUDAInt(stateDict.DilationH, stateDict.Dilation),
		defaultCUDAInt(stateDict.DilationW, stateDict.Dilation),
		defaultCUDAInt(stateDict.Groups, 1), stateDict.OutH, stateDict.OutW,
	)

	return setCUDAOutput(stateDict, output, err)
}

type MaxPool2D struct{ pooling *CUDAPooling }
type AvgPool2D struct{ pooling *CUDAPooling }
type AdaptiveAvgPool2D struct{ pooling *CUDAPooling }
type AdaptiveMaxPool2D struct{ pooling *CUDAPooling }

func (pooling *MaxPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.pooling.max_pool2d"); err != nil {
		return nil, err
	}

	output, err := pooling.pooling.MaxPool2d(
		stateDict.OperationShape(), cudaMaxPoolParams(stateDict), stateDict.Inputs[0],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (pooling *AvgPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.pooling.avg_pool2d"); err != nil {
		return nil, err
	}

	output, err := pooling.pooling.AvgPool2d(
		stateDict.OperationShape(), cudaAvgPoolParams(stateDict), stateDict.Inputs[0],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (pooling *AdaptiveAvgPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.pooling.adaptive_avg_pool2d"); err != nil {
		return nil, err
	}

	output, err := pooling.pooling.AdaptiveAvgPool2d(
		stateDict.OperationShape(), stateDict.OutH, stateDict.OutW, stateDict.Inputs[0],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (pooling *AdaptiveMaxPool2D) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.pooling.adaptive_max_pool2d"); err != nil {
		return nil, err
	}

	output, err := pooling.pooling.AdaptiveMaxPool2d(
		stateDict.OperationShape(), stateDict.OutH, stateDict.OutW, stateDict.Inputs[0],
	)

	return setCUDAOutput(stateDict, output, err)
}

type Linear struct{ projection *CUDAProjection }
type FusedQKV struct{ projection *CUDAProjection }

func (linear *Linear) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.projection.linear"); err != nil {
		return nil, err
	}

	weight, bias := cudaWeightBias(stateDict)
	output, err := linear.projection.Linear(
		stateDict.OperationShape(), weight, bias, stateDict.Inputs[0],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (fusedQKV *FusedQKV) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.projection.fused_qkv"); err != nil {
		return nil, err
	}

	weight, bias := cudaWeightBias(stateDict)
	output, err := fusedQKV.projection.FusedQKV(
		stateDict.OperationShape(), weight, bias, stateDict.Inputs[0],
	)

	return setCUDAOutput(stateDict, output, err)
}

type HawkesIntensity struct{ hawkes *CUDAHawkes }
type HawkesKernelMatrix struct{ hawkes *CUDAHawkes }
type HawkesLogLikelihood struct{ hawkes *CUDAHawkes }
type HawkesSimulate struct{ hawkes *CUDAHawkes }

func (hawkes *HawkesIntensity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.hawkes.intensity", 5, hawkes.hawkes.Intensity)
}

func (hawkes *HawkesKernelMatrix) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.hawkes.kernel_matrix", 3, hawkes.hawkes.KernelMatrix,
	)
}

func (hawkes *HawkesLogLikelihood) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.hawkes.log_likelihood", 3, hawkes.hawkes.LogLikelihood,
	)
}

func (hawkes *HawkesSimulate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.hawkes.simulate", 4, hawkes.hawkes.Simulate)
}

type VSABind struct{ vsaOps *CUDAVSAOps }
type VSABundle struct{ vsaOps *CUDAVSAOps }
type VSASimilarity struct{ vsaOps *CUDAVSAOps }
type VSAPermute struct{ vsaOps *CUDAVSAOps }
type VSAInversePermute struct{ vsaOps *CUDAVSAOps }

func (vsaBind *VSABind) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.vsa.bind", 2, vsaBind.vsaOps.Bind)
}

func (vsaBundle *VSABundle) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.vsa.bundle", 1, vsaBundle.vsaOps.Bundle)
}

func (vsaSimilarity *VSASimilarity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.vsa.similarity", 2, vsaSimilarity.vsaOps.Similarity)
}

func (vsaPermute *VSAPermute) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.vsa.permute"); err != nil {
		return nil, err
	}

	output, err := vsaPermute.vsaOps.Permute(
		stateDict.OperationShape(), stateDict.Dim, stateDict.Inputs...,
	)

	return setCUDAOutput(stateDict, output, err)
}

func (vsaInversePermute *VSAInversePermute) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("cuda.vsa.inverse_permute"); err != nil {
		return nil, err
	}

	output, err := vsaInversePermute.vsaOps.InversePermute(
		stateDict.OperationShape(), stateDict.Dim, stateDict.Inputs...,
	)

	return setCUDAOutput(stateDict, output, err)
}

type BeliefUpdate struct{ activeInference *CUDAActiveInferenceOps }
type ExpectedFreeEnergy struct{ activeInference *CUDAActiveInferenceOps }
type FreeEnergy struct{ activeInference *CUDAActiveInferenceOps }
type PrecisionWeight struct{ activeInference *CUDAActiveInferenceOps }

func (activeInference *BeliefUpdate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.active_inference.belief_update", 3,
		activeInference.activeInference.BeliefUpdate,
	)
}

func (activeInference *ExpectedFreeEnergy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.active_inference.expected_free_energy", 1,
		activeInference.activeInference.ExpectedFreeEnergy,
	)
}

func (activeInference *FreeEnergy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.active_inference.free_energy", 2,
		activeInference.activeInference.FreeEnergy,
	)
}

func (activeInference *PrecisionWeight) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.active_inference.precision_weight", 2,
		activeInference.activeInference.PrecisionWeight,
	)
}

type Prediction struct{ predictiveCoding *CUDAPredictiveCodingOps }
type PredictionError struct{ predictiveCoding *CUDAPredictiveCodingOps }
type UpdateRepresentation struct{ predictiveCoding *CUDAPredictiveCodingOps }
type UpdateWeights struct{ predictiveCoding *CUDAPredictiveCodingOps }

func (prediction *Prediction) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.predictive_coding.prediction", 2,
		prediction.predictiveCoding.Prediction,
	)
}

func (predictionError *PredictionError) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.predictive_coding.prediction_error", 2,
		predictionError.predictiveCoding.PredictionError,
	)
}

func (updateRepresentation *UpdateRepresentation) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.predictive_coding.update_representation", 5,
		updateRepresentation.predictiveCoding.UpdateRepresentation,
	)
}

func (updateWeights *UpdateWeights) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.predictive_coding.update_weights", 4,
		updateWeights.predictiveCoding.UpdateWeights,
	)
}

type FlowActive struct{ markovBlanket *CUDAMarkovBlanket }
type FlowInternal struct{ markovBlanket *CUDAMarkovBlanket }
type MutualInformation struct{ markovBlanket *CUDAMarkovBlanket }
type Partition struct{ markovBlanket *CUDAMarkovBlanket }

func (markovBlanket *FlowActive) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.markov_blanket.flow_active", 3,
		markovBlanket.markovBlanket.FlowActive,
	)
}

func (markovBlanket *FlowInternal) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.markov_blanket.flow_internal", 3,
		markovBlanket.markovBlanket.FlowInternal,
	)
}

func (markovBlanket *MutualInformation) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.markov_blanket.mutual_information", 2,
		markovBlanket.markovBlanket.MutualInformation,
	)
}

func (markovBlanket *Partition) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.markov_blanket.partition", 2,
		markovBlanket.markovBlanket.Partition,
	)
}

type BackdoorAdjustment struct{ causal *CUDACausalOps }
type CATE struct{ causal *CUDACausalOps }
type Counterfactual struct{ causal *CUDACausalOps }
type DAGMarkovFactorization struct{ causal *CUDACausalOps }
type DoCalculus struct{ causal *CUDACausalOps }
type FrontdoorAdjustment struct{ causal *CUDACausalOps }
type IVEstimate struct{ causal *CUDACausalOps }

func (causal *BackdoorAdjustment) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.causal.backdoor_adjustment", 3, causal.causal.BackdoorAdjustment,
	)
}

func (causal *CATE) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.causal.cate", 3, causal.causal.CATE)
}

func (causal *Counterfactual) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.causal.counterfactual", 4, causal.causal.Counterfactual,
	)
}

func (causal *DAGMarkovFactorization) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.causal.dag_markov_factorization", 2,
		causal.causal.DAGMarkovFactorization,
	)
}

func (causal *DoCalculus) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.causal.do_calculus", 3, causal.causal.DoCalculus)
}

func (causal *FrontdoorAdjustment) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(
		stateDict, "cuda.causal.frontdoor_adjustment", 3,
		causal.causal.FrontdoorAdjustment,
	)
}

func (causal *IVEstimate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	return cudaShapeForward(stateDict, "cuda.causal.iv_estimate", 3, causal.causal.IVEstimate)
}

type MSELoss struct{ trainingOps *CUDATrainingOps }
type CrossEntropyLoss struct{ trainingOps *CUDATrainingOps }
type MSEGrad struct{ trainingOps *CUDATrainingOps }
type CrossEntropyGrad struct{ trainingOps *CUDATrainingOps }
type Accuracy struct{ trainingOps *CUDATrainingOps }
type Perplexity struct{ trainingOps *CUDATrainingOps }
type F1 struct{ trainingOps *CUDATrainingOps }

func (mseLoss *MSELoss) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := cudaTrainInputs(stateDict, "cuda.train.mse_loss"); err != nil {
		return nil, err
	}

	output, err := mseLoss.trainingOps.MSELoss(stateDict.Inputs[0], stateDict.Inputs[1])
	return setCUDAOutput(stateDict, output, err)
}

func (crossEntropyLoss *CrossEntropyLoss) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	if err := cudaTrainInputs(stateDict, "cuda.train.cross_entropy_loss"); err != nil {
		return nil, err
	}

	output, err := crossEntropyLoss.trainingOps.CrossEntropyLoss(
		stateDict.Inputs[0], stateDict.Inputs[1],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (mseGrad *MSEGrad) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := cudaTrainInputs(stateDict, "cuda.train.mse_grad"); err != nil {
		return nil, err
	}

	output, err := mseGrad.trainingOps.MSEGrad(stateDict.Inputs[0], stateDict.Inputs[1])
	return setCUDAOutput(stateDict, output, err)
}

func (crossEntropyGrad *CrossEntropyGrad) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	if err := cudaTrainInputs(stateDict, "cuda.train.cross_entropy_grad"); err != nil {
		return nil, err
	}

	output, err := crossEntropyGrad.trainingOps.CrossEntropyGrad(
		stateDict.Inputs[0], stateDict.Inputs[1],
	)

	return setCUDAOutput(stateDict, output, err)
}

func (accuracy *Accuracy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := cudaTrainInputs(stateDict, "cuda.bench.accuracy"); err != nil {
		return nil, err
	}

	output, err := accuracy.trainingOps.Accuracy(stateDict.Inputs[0], stateDict.Inputs[1])

	if err != nil {
		return nil, err
	}

	stateDict.Total++

	if output[0] == 1 {
		stateDict.Correct++
	}

	stateDict.EnsureOperationOutLen(1)
	stateDict.Out[0] = float64(stateDict.Correct) / float64(stateDict.Total)

	return stateDict, nil
}

func (perplexity *Perplexity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := cudaTrainInputs(stateDict, "cuda.bench.perplexity"); err != nil {
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
	if err := cudaTrainInputs(stateDict, "cuda.bench.f1"); err != nil {
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

func cudaUnaryForward(
	stateDict *state.Dict,
	name string,
	forward func([]float64) ([]float64, error),
) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs(name, 1); err != nil {
		return nil, err
	}

	output, err := forward(stateDict.Inputs[0])
	return setCUDAOutput(stateDict, output, err)
}

func cudaShapeForward(
	stateDict *state.Dict,
	name string,
	inputCount int,
	forward func([]int, ...[]float64) ([]float64, error),
) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs(name, inputCount); err != nil {
		return nil, err
	}

	output, err := forward(stateDict.OperationShape(), stateDict.Inputs...)
	return setCUDAOutput(stateDict, output, err)
}

func cudaTrainInputs(stateDict *state.Dict, name string) error {
	if err := stateDict.RequireOperationInputs(name, 2); err != nil {
		return err
	}

	if len(stateDict.Inputs[0]) == 0 || len(stateDict.Inputs[0]) != len(stateDict.Inputs[1]) {
		return fmt.Errorf("%s: input lengths must match and be non-zero", name)
	}

	return nil
}

func setCUDAOutput(
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

func requireCUDAShape(
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

func cudaLastTokenShapeParts(shape []int) (int, int, int, error) {
	if len(shape) < 2 {
		return 0, 0, 0, fmt.Errorf("cuda.shape.last_token: expected rank >= 2")
	}

	sequenceLength := shape[len(shape)-2]
	featureLength := shape[len(shape)-1]

	if sequenceLength <= 0 || featureLength <= 0 {
		return 0, 0, 0, fmt.Errorf("cuda.shape.last_token: trailing dimensions must be positive")
	}

	outer := 1

	for _, dimension := range shape[:len(shape)-2] {
		if dimension <= 0 {
			return 0, 0, 0, fmt.Errorf("cuda.shape.last_token: outer dimensions must be positive")
		}

		if outer > stdmath.MaxInt/dimension {
			return 0, 0, 0, fmt.Errorf("cuda.shape.last_token: shape product overflows int")
		}

		outer *= dimension
	}

	return outer, sequenceLength, featureLength, nil
}

func cudaWeightBias(stateDict *state.Dict) ([]float64, []float64) {
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

func cudaMaxPoolParams(stateDict *state.Dict) CUDAMaxPool2dParams {
	return CUDAMaxPool2dParams{
		KernelH:   defaultCUDAInt(stateDict.KernelH, stateDict.KernelSize),
		KernelW:   defaultCUDAInt(stateDict.KernelW, stateDict.KernelSize),
		StrideH:   defaultCUDAInt(stateDict.StrideH, stateDict.Stride),
		StrideW:   defaultCUDAInt(stateDict.StrideW, stateDict.Stride),
		PadH:      stateDict.PadH,
		PadW:      stateDict.PadW,
		DilationH: defaultCUDAInt(stateDict.DilationH, stateDict.Dilation),
		DilationW: defaultCUDAInt(stateDict.DilationW, stateDict.Dilation),
		CeilMode:  stateDict.Ceil,
	}
}

func cudaAvgPoolParams(stateDict *state.Dict) CUDAAvgPool2dParams {
	return CUDAAvgPool2dParams{
		KernelH:         defaultCUDAInt(stateDict.KernelH, stateDict.KernelSize),
		KernelW:         defaultCUDAInt(stateDict.KernelW, stateDict.KernelSize),
		StrideH:         defaultCUDAInt(stateDict.StrideH, stateDict.Stride),
		StrideW:         defaultCUDAInt(stateDict.StrideW, stateDict.Stride),
		PadH:            stateDict.PadH,
		PadW:            stateDict.PadW,
		DilationH:       defaultCUDAInt(stateDict.DilationH, stateDict.Dilation),
		DilationW:       defaultCUDAInt(stateDict.DilationW, stateDict.Dilation),
		CeilMode:        stateDict.Ceil,
		CountIncludePad: stateDict.CountPad,
		DivisorOverride: stateDict.Divisor,
	}
}

func cudaStateConfig(config *state.Dict) *state.Dict {
	if config == nil {
		return state.NewDict()
	}

	return config
}

func defaultCUDAInt(value, fallback int) int {
	if value == 0 {
		return fallback
	}

	return value
}

func defaultCUDAFloat(value, fallback float64) float64 {
	if value == 0 {
		return fallback
	}

	return value
}
