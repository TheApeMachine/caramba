package tuner

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/active_inference"
)

/*
Arm represents a single candidate in the hyperparameter tuning or
data selection process.
*/
type Arm struct {
	ID     string
	Config map[string]any
}

/*
Bandit provides a multi-armed bandit interface for optimizing the training pipeline.
When configured with strategy="active_inference", it uses Expected Free Energy
to balance exploration and exploitation across the available arms.
*/
type Bandit struct {
	strategy string
	arms     []Arm
	efe      *active_inference.ExpectedFreeEnergy
}

/*
NewBandit instantiates a multi-armed bandit tuner.
*/
func NewBandit(strategy string, arms []Arm) (*Bandit, error) {
	if len(arms) == 0 {
		return nil, fmt.Errorf("tuner: cannot create bandit with 0 arms")
	}

	return &Bandit{
		strategy: strategy,
		arms:     arms,
		efe:      active_inference.NewExpectedFreeEnergy(),
	}, nil
}

/*
SelectArm chooses the optimal arm given the state probabilities for each arm.
stateProbs must be a 1D slice of size N * K, where N is the number of outcome
states and K is the number of arms.
Returns the selected Arm and its corresponding Expected Free Energy score.
*/
func (bandit *Bandit) SelectArm(stateProbs []float64, numStates int) (Arm, float64, error) {
	numArms := len(bandit.arms)

	if len(stateProbs) != numStates*numArms {
		return Arm{}, 0.0, fmt.Errorf(
			"tuner: stateProbs length %d must equal numStates*numArms (%d*%d=%d)",
			len(stateProbs), numStates, numArms, numStates*numArms,
		)
	}

	if bandit.strategy == "active_inference" {
		efeValues := bandit.efe.Forward([]int{numStates, numArms}, stateProbs)

		bestIdx := 0
		minEFE := math.MaxFloat64

		for i, efe := range efeValues {
			if efe < minEFE {
				minEFE = efe
				bestIdx = i
			}
		}

		return bandit.arms[bestIdx], minEFE, nil
	}

	// Fallback to random or basic greedy if strategy is not active_inference.
	// For simplicity, we just return the first arm if not using active_inference.
	return bandit.arms[0], 0.0, nil
}
