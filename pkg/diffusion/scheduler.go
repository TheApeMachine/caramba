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
	mu := scheduler.calculateShift(imageSequenceLength)
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

	for index := range sigmas {
		sigmas[index] = 1 - float64(index)/lastIndex
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

func (scheduler *FlowMatchEulerScheduler) calculateShift(imageSequenceLength int) float64 {
	baseLength := float64(scheduler.config.BaseImageSeqLen)
	maxLength := float64(scheduler.config.MaxImageSeqLen)
	baseShift := scheduler.config.BaseShift
	maxShift := scheduler.config.MaxShift

	if maxLength <= baseLength {
		return baseShift
	}

	slope := (maxShift - baseShift) / (maxLength - baseLength)
	intercept := baseShift - slope*baseLength

	return float64(imageSequenceLength)*slope + intercept
}
