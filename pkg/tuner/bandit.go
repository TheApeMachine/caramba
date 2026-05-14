package tuner

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/active_inference"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

const banditNumStates = 8

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
	strategy  string
	arms      []Arm
	efe       *active_inference.ExpectedFreeEnergy
	posterior []float64
	means     []float64
	m2        []float64
	counts    []int
	numStates int
	random    *rand.Rand
}

/*
NewBandit instantiates a multi-armed bandit tuner.
*/
func NewBandit(strategy string, arms []Arm, seed int64) (*Bandit, error) {
	if len(arms) == 0 {
		return nil, fmt.Errorf("tuner: cannot create bandit with 0 arms")
	}

	if strategy != "active_inference" {
		return nil, fmt.Errorf("tuner: unsupported bandit strategy %q", strategy)
	}

	posterior := make([]float64, len(arms))
	uniform := 1.0 / float64(len(arms))

	for index := range posterior {
		posterior[index] = uniform
	}

	return &Bandit{
		strategy:  strategy,
		arms:      arms,
		efe:       active_inference.NewExpectedFreeEnergy(),
		posterior: posterior,
		means:     make([]float64, len(arms)),
		m2:        make([]float64, len(arms)),
		counts:    make([]int, len(arms)),
		numStates: banditNumStates,
		random:    rand.New(rand.NewSource(seed)),
	}, nil
}

/*
SelectArm chooses the next arm to evaluate.
Returns the selected Arm and its corresponding Expected Free Energy score.
*/
func (bandit *Bandit) SelectArm() (Arm, float64, error) {
	for index, count := range bandit.counts {
		if count == 0 {
			return bandit.arms[index], bandit.score(index), nil
		}
	}

	threshold := bandit.random.Float64()
	cumulative := 0.0

	for index, probability := range bandit.posterior {
		cumulative += probability

		if threshold <= cumulative {
			return bandit.arms[index], bandit.score(index), nil
		}
	}

	lastIndex := len(bandit.arms) - 1

	return bandit.arms[lastIndex], bandit.score(lastIndex), nil
}

/*
Update records the observed metric for the most recent trial of armID.
Lower metric is better.
*/
func (bandit *Bandit) Update(armID string, metric float64) error {
	index, err := bandit.index(armID)

	if err != nil {
		return err
	}

	bandit.counts[index]++
	delta := metric - bandit.means[index]
	bandit.means[index] += delta / float64(bandit.counts[index])
	bandit.m2[index] += delta * (metric - bandit.means[index])
	bandit.recomputePosterior()

	return nil
}

func (bandit *Bandit) index(armID string) (int, error) {
	for index, arm := range bandit.arms {
		if arm.ID == armID {
			return index, nil
		}
	}

	return 0, fmt.Errorf("tuner: unknown arm ID %q", armID)
}

func (bandit *Bandit) score(index int) float64 {
	return -math.Log(bandit.posterior[index])
}

func (bandit *Bandit) recomputePosterior() {
	stateProbs := bandit.stateProbabilities()
	stateDict, err := bandit.efe.Forward(
		state.NewDict().
			WithShape([]int{bandit.numStates, len(bandit.arms)}).
			WithInput(stateProbs),
	)

	if err != nil {
		panic(err)
	}

	efeValues := stateDict.Out
	bestIndex := bandit.bestObservedIndex()

	for index := range efeValues {
		efeValues[index] += bandit.means[index] - bandit.means[bestIndex]
	}

	bandit.softmaxNegate(efeValues)
}

func (bandit *Bandit) stateProbabilities() []float64 {
	numArms := len(bandit.arms)
	stateProbs := make([]float64, bandit.numStates*numArms)
	minMetric, maxMetric := bandit.metricRange()

	for armIndex, count := range bandit.counts {
		if count == 0 {
			bandit.distributeUniform(stateProbs, armIndex)

			continue
		}

		bucket := bandit.bucket(bandit.means[armIndex], minMetric, maxMetric)
		stateProbs[bucket*numArms+armIndex] = 1.0
	}

	return stateProbs
}

func (bandit *Bandit) metricRange() (float64, float64) {
	minMetric := math.MaxFloat64
	maxMetric := -math.MaxFloat64

	for index, count := range bandit.counts {
		if count == 0 {
			continue
		}

		minMetric = math.Min(minMetric, bandit.means[index])
		maxMetric = math.Max(maxMetric, bandit.means[index])
	}

	if minMetric == math.MaxFloat64 {
		return 0.0, 0.0
	}

	return minMetric, maxMetric
}

func (bandit *Bandit) bucket(metric float64, minMetric float64, maxMetric float64) int {
	if minMetric == maxMetric {
		return 0
	}

	ratio := (metric - minMetric) / (maxMetric - minMetric)
	bucket := int(ratio * float64(bandit.numStates-1))

	return min(max(bucket, 0), bandit.numStates-1)
}

func (bandit *Bandit) distributeUniform(stateProbs []float64, armIndex int) {
	numArms := len(bandit.arms)
	probability := 1.0 / float64(bandit.numStates)

	for bucket := range bandit.numStates {
		stateProbs[bucket*numArms+armIndex] = probability
	}
}

func (bandit *Bandit) bestObservedIndex() int {
	bestIndex := 0
	bestMetric := math.MaxFloat64

	for index, count := range bandit.counts {
		if count == 0 || bandit.means[index] >= bestMetric {
			continue
		}

		bestMetric = bandit.means[index]
		bestIndex = index
	}

	return bestIndex
}

func (bandit *Bandit) softmaxNegate(values []float64) {
	maxValue := -math.MaxFloat64

	for _, value := range values {
		maxValue = math.Max(maxValue, -value)
	}

	total := 0.0

	for index, value := range values {
		bandit.posterior[index] = math.Exp(-value - maxValue)
		total += bandit.posterior[index]
	}

	for index := range bandit.posterior {
		bandit.posterior[index] /= total
	}
}
