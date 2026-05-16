package dispatch

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestBuildOperation(t *testing.T) {
	Convey("Given the native dispatch contract", t, func() {
		operations := operationRegistry{}
		optimizers := optimizerRegistry{}

		Convey("It should bind every required executable operation", func() {
			for _, operationID := range ir.RequiredOperationIDs() {
				switch executor.NormalizeOperation(operationID) {
				case ir.OpInput, ir.OpFused:
					continue
				}

				node := executor.NodeSpec{Op: operationID}

				if IsOptimizerOperation(operationID) {
					optimizer, err := BuildOptimizer(optimizers, node)

					So(err, ShouldBeNil)
					So(optimizer, ShouldNotBeNil)

					continue
				}

				operation, err := BuildOperation(operations, node)

				So(err, ShouldBeNil)
				So(operation, ShouldNotBeNil)
			}
		})
	})
}

type fakeOperation struct{}

func (operation fakeOperation) Forward(stateDict *state.Dict) (*state.Dict, error) {
	stateDict.SetOperationOutput([]float64{1})

	return stateDict, nil
}

type fakeOptimizer struct{}

func (optimizer fakeOptimizer) Step(stateDict *state.Dict) (*state.Dict, error) {
	stateDict.SetOperationOutput([]float64{1})

	return stateDict, nil
}

type operationRegistry struct{}

func (registry operationRegistry) operation(*state.Dict) (state.Operation, error) {
	return fakeOperation{}, nil
}

func (registry operationRegistry) ReLU(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) LeakyReLU(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) GELU(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Tanh(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Sigmoid(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) SwiGLU(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Swish(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) SELU(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) SDPA(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) MQA(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) GQA(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) SlidingWindowAttention(
	config *state.Dict,
) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) ApplyMask(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) CausalMask(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Add(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Mul(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Matmul(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Exp(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Log(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) LogSumExp(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Softmax(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Outer(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Sign(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) InvSqrtDimScale(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Dropout(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) RMSNorm(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) LayerNorm(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) GroupNorm(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Reshape(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Transpose(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Concat(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Split(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) UpsampleNearest2D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) ViewAsHeads(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) MergeHeads(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) LastToken(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) RoPE(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) ALiBi(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) TokenEmbedding(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Conv1D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Conv2D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Conv3D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) ConvTranspose2D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) MaxPool2D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) AvgPool2D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) AdaptiveAvgPool2D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) AdaptiveMaxPool2D(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Linear(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) FusedQKV(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) HawkesIntensity(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) HawkesKernelMatrix(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) HawkesLogLikelihood(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) HawkesSimulate(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) VSABind(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) VSABundle(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) VSASimilarity(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) VSAPermute(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) VSAInversePermute(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) BeliefUpdate(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) ExpectedFreeEnergy(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) FreeEnergy(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) PrecisionWeight(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Prediction(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) PredictionError(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) UpdateRepresentation(
	config *state.Dict,
) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) UpdateWeights(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) FlowActive(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) FlowInternal(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) MutualInformation(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Partition(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) BackdoorAdjustment(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) CATE(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Counterfactual(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) DAGMarkovFactorization(
	config *state.Dict,
) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) DoCalculus(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) FrontdoorAdjustment(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) IVEstimate(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) MSELoss(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) CrossEntropyLoss(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) MSEGrad(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) CrossEntropyGrad(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Accuracy(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Perplexity(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) F1(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Graft(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

func (registry operationRegistry) Freeze(config *state.Dict) (state.Operation, error) {
	return registry.operation(config)
}

type optimizerRegistry struct{}

func (registry optimizerRegistry) Adam(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) AdamW(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) AdaMax(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) SGD(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) Lion(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) RMSProp(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) Hebbian(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) Lars(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) Lamb(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) AdaGrad(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) AdaDelta(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}

func (registry optimizerRegistry) LBFGS(*state.Dict) (state.Optimizer, error) {
	return fakeOptimizer{}, nil
}
