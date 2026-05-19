package tuner

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
ExpectedFreeEnergy computes epistemic expected free energy per outcome arm:

	G[k] = -sum_i q_outcomes[i,k] * ln(q_outcomes[i,k] + eps)

shape = [N, K]: N state buckets, K arms. Input is row-major q_outcomes.
*/
type ExpectedFreeEnergy struct{}

/*
NewExpectedFreeEnergy instantiates the bandit EFE scorer.
*/
func NewExpectedFreeEnergy() *ExpectedFreeEnergy {
	return &ExpectedFreeEnergy{}
}

/*
Forward scores each outcome column of q_outcomes into stateDict.Out.
*/
func (expectedFreeEnergy *ExpectedFreeEnergy) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	numStates, numOutcomes, err := expectedFreeEnergy.validateForwardInput(stateDict)

	if err != nil {
		return nil, err
	}

	stateDict.EnsureOperationOutLen(numOutcomes)
	expectedFreeEnergy.writeOutcomeScores(
		stateDict.Out,
		stateDict.Inputs[0],
		numStates,
		numOutcomes,
	)

	return stateDict, nil
}

func (expectedFreeEnergy *ExpectedFreeEnergy) validateForwardInput(
	stateDict *state.Dict,
) (int, int, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return 0, 0, fmt.Errorf(
			"tuner.expected_free_energy: len(shape)=%d, need >= 2",
			len(shape),
		)
	}

	numStates, numOutcomes := shape[0], shape[1]

	if numStates <= 0 || numOutcomes <= 0 {
		return 0, 0, fmt.Errorf(
			"tuner.expected_free_energy: need shape[0]=N > 0 and shape[1]=K > 0 (got N=%d K=%d)",
			numStates, numOutcomes,
		)
	}

	if err := stateDict.RequireOperation("tuner.expected_free_energy"); err != nil {
		return 0, 0, err
	}

	if len(stateDict.Inputs[0]) != numStates*numOutcomes {
		return 0, 0, fmt.Errorf(
			"tuner.expected_free_energy: len(q_outcomes)=%d, need N*K=%d",
			len(stateDict.Inputs[0]), numStates*numOutcomes,
		)
	}

	for index, value := range stateDict.Inputs[0] {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, 0, fmt.Errorf(
				"tuner.expected_free_energy: q_outcomes[%d]=%g is not finite",
				index, value,
			)
		}
	}

	return numStates, numOutcomes, nil
}

func (expectedFreeEnergy *ExpectedFreeEnergy) writeOutcomeScores(
	out []float64,
	qOutcomes []float64,
	numStates int,
	numOutcomes int,
) {
	const epsilon = 1e-12

	for outcomeIndex := range numOutcomes {
		var sum float64

		for stateIndex := range numStates {
			probability := qOutcomes[stateIndex*numOutcomes+outcomeIndex]

			if probability < 0 {
				probability = 0
			}

			if probability > 1 {
				probability = 1
			}

			sum += probability * math.Log(probability+epsilon)
		}

		out[outcomeIndex] = -sum
	}
}
