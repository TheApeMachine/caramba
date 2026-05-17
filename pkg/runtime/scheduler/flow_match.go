package scheduler

import (
	"context"
	"fmt"
	"sync"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
FlowMatchEuler implements op.SchedulerRunner for the discrete
flow-match Euler scheduler used by manifest-declared diffusion
programs.

State is cached per scheduler-id so Timesteps and Step share the same
sigma schedule; cache entries are rebuilt when the declared config
changes.
*/
type FlowMatchEuler struct {
	mu     sync.Mutex
	cached map[string]*schedule
}

func NewFlowMatchEuler() *FlowMatchEuler {
	return &FlowMatchEuler{cached: map[string]*schedule{}}
}

type schedule struct {
	configFingerprint string
	timesteps         []float64
	sigmas            []float64
}

func (flowMatch *FlowMatchEuler) Timesteps(
	execContext context.Context,
	declaration program.SchedulerDeclaration,
) ([]float64, error) {
	cached, err := flowMatch.cacheFor(declaration)

	if err != nil {
		return nil, err
	}

	return append([]float64(nil), cached.timesteps...), nil
}

func (flowMatch *FlowMatchEuler) Step(
	execContext context.Context,
	declaration program.SchedulerDeclaration,
	stepIndex int,
	latents []float64,
	modelOutput []float64,
) ([]float64, error) {
	cached, err := flowMatch.cacheFor(declaration)

	if err != nil {
		return nil, err
	}

	if stepIndex < 0 || stepIndex+1 >= len(cached.sigmas) {
		return nil, fmt.Errorf("scheduler/flow_match: step %d out of range", stepIndex)
	}

	if len(latents) != len(modelOutput) {
		return nil, fmt.Errorf(
			"scheduler/flow_match: model output length %d != latents length %d",
			len(modelOutput),
			len(latents),
		)
	}

	delta := cached.sigmas[stepIndex+1] - cached.sigmas[stepIndex]
	next := make([]float64, len(latents))

	for index, latent := range latents {
		next[index] = latent + delta*modelOutput[index]
	}

	return next, nil
}

func (flowMatch *FlowMatchEuler) cacheFor(
	declaration program.SchedulerDeclaration,
) (*schedule, error) {
	if declaration.Type != "" && declaration.Type != "flow_match_euler_discrete" && declaration.Type != "flow_match_euler" {
		return nil, fmt.Errorf("scheduler/flow_match: unsupported type %q", declaration.Type)
	}

	fingerprint := fingerprintConfig(declaration.Config)

	flowMatch.mu.Lock()
	defer flowMatch.mu.Unlock()

	if cached, ok := flowMatch.cached[declaration.ID]; ok && cached.configFingerprint == fingerprint {
		return cached, nil
	}

	configuration, err := parseFlowMatchConfig(declaration.Config)

	if err != nil {
		return nil, err
	}

	sigmas := buildSigmas(configuration)
	timesteps := make([]float64, len(sigmas)-1)

	for index := range timesteps {
		timesteps[index] = sigmas[index] * float64(configuration.numTrainTimesteps)
	}

	cached := &schedule{
		configFingerprint: fingerprint,
		timesteps:         timesteps,
		sigmas:            sigmas,
	}

	flowMatch.cached[declaration.ID] = cached

	return cached, nil
}
