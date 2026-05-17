package diffusion

import (
	"fmt"
	"math"
)

type FlowMatchEulerScheduler struct {
	config    SchedulerConfig
	timesteps []float64
	sigmas    []float64
}

func NewFlowMatchEulerScheduler(config SchedulerConfig, imageSequenceLength int) (*FlowMatchEulerScheduler, error) {
	if config.Type != "" && config.Type != "flow_match_euler_discrete" {
		return nil, fmt.Errorf("diffusion: unsupported scheduler %q", config.Type)
	}

	if config.Steps <= 0 {
		return nil, fmt.Errorf("diffusion: scheduler steps must be positive")
	}

	if config.NumTrainTimesteps <= 0 {
		return nil, fmt.Errorf("diffusion: scheduler num_train_timesteps must be positive")
	}

	switch config.TimeShiftType {
	case "", "exponential", "linear":
	default:
		return nil, fmt.Errorf("diffusion: unsupported scheduler time_shift_type %q", config.TimeShiftType)
	}

	scheduler := &FlowMatchEulerScheduler{config: config}
	mu := scheduler.empiricalShift(imageSequenceLength)
	sigmas := scheduler.linearSigmas()

	if config.UseDynamicShift {
		sigmas = scheduler.timeShift(mu, sigmas)
	} else {
		sigmas = scheduler.staticShift(sigmas)
	}

	scheduler.sigmas = append(sigmas, 0)
	scheduler.timesteps = make([]float64, len(sigmas))

	for index, sigma := range sigmas {
		scheduler.timesteps[index] = sigma * float64(config.NumTrainTimesteps)
	}

	return scheduler, nil
}

func (scheduler *FlowMatchEulerScheduler) Timesteps() []float64 {
	return append([]float64(nil), scheduler.timesteps...)
}

func (scheduler *FlowMatchEulerScheduler) Step(
	stepIndex int,
	latents []float64,
	modelOutput []float64,
) ([]float64, error) {
	if stepIndex < 0 || stepIndex+1 >= len(scheduler.sigmas) {
		return nil, fmt.Errorf("diffusion: scheduler step %d out of range", stepIndex)
	}

	if len(latents) != len(modelOutput) {
		return nil, fmt.Errorf(
			"diffusion: model output length %d does not match latent length %d",
			len(modelOutput),
			len(latents),
		)
	}

	delta := scheduler.sigmas[stepIndex+1] - scheduler.sigmas[stepIndex]
	next := make([]float64, len(latents))

	for index, latent := range latents {
		next[index] = latent + delta*modelOutput[index]
	}

	return next, nil
}

func (scheduler *FlowMatchEulerScheduler) linearSigmas() []float64 {
	if scheduler.config.Steps == 1 {
		return []float64{1}
	}

	sigmas := make([]float64, scheduler.config.Steps)
	lastIndex := float64(scheduler.config.Steps - 1)
	minSigma := 1 / float64(scheduler.config.Steps)

	for index := range sigmas {
		fraction := float64(index) / lastIndex
		sigmas[index] = 1 - fraction*(1-minSigma)
	}

	return sigmas
}

func (scheduler *FlowMatchEulerScheduler) staticShift(sigmas []float64) []float64 {
	out := make([]float64, len(sigmas))
	shift := scheduler.config.Shift

	for index, sigma := range sigmas {
		out[index] = shift * sigma / (1 + (shift-1)*sigma)
	}

	return out
}

func (scheduler *FlowMatchEulerScheduler) timeShift(mu float64, sigmas []float64) []float64 {
	out := make([]float64, len(sigmas))

	for index, sigma := range sigmas {
		switch scheduler.config.TimeShiftType {
		case "", "exponential":
			out[index] = math.Exp(mu) / (math.Exp(mu) + math.Pow(1/sigma-1, 1))
		case "linear":
			out[index] = mu / (mu + math.Pow(1/sigma-1, 1))
		}
	}

	return out
}

func (scheduler *FlowMatchEulerScheduler) empiricalShift(imageSequenceLength int) float64 {
	sequenceLength := float64(imageSequenceLength)
	stepCount := float64(scheduler.config.Steps)

	highStepSlope := 0.00016927
	highStepIntercept := 0.45666666

	if imageSequenceLength > 4300 {
		return highStepSlope*sequenceLength + highStepIntercept
	}

	lowStepSlope := 8.73809524e-05
	lowStepIntercept := 1.89833333
	highStepMu := highStepSlope*sequenceLength + highStepIntercept
	lowStepMu := lowStepSlope*sequenceLength + lowStepIntercept
	slope := (highStepMu - lowStepMu) / 190
	intercept := highStepMu - 200*slope

	return slope*stepCount + intercept
}
