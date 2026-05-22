package diffusion

import (
	"context"
	"fmt"
	"io"

	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
)

func (harness *Harness) baselineTensors() (tensor.Tensor, tensor.Tensor, error) {
	latents, err := harness.stateTensor("state.latents")

	if err != nil {
		return nil, nil, err
	}

	promptEmbeds, err := harness.stateTensor("state.prompt_embeds")

	if err != nil {
		return nil, nil, err
	}

	latentClone, err := CloneTensor(harness.StateMemory, latents)

	if err != nil {
		return nil, nil, err
	}

	promptClone, err := CloneTensor(harness.StateMemory, promptEmbeds)

	if err != nil {
		latentClone.Close()
		return nil, nil, err
	}

	return latentClone, promptClone, nil
}

func (harness *Harness) stateTensor(reference string) (tensor.Tensor, error) {
	store := harness.Session.StateStore()

	if store == nil {
		return nil, fmt.Errorf("diffusion diagnose: state store is required")
	}

	value, err := store.ResolveReference(reference)

	if err != nil {
		return nil, err
	}

	resident, ok := value.(tensor.Tensor)

	if !ok {
		return nil, fmt.Errorf("diffusion diagnose: %q is %T, expected tensor.Tensor", reference, value)
	}

	return resident, nil
}

func (harness *Harness) denoiseVelocity(
	ctx context.Context,
	latents tensor.Tensor,
	promptEmbeds tensor.Tensor,
	timestep float32,
) ([]float32, error) {
	result, err := harness.Session.CallGraph(ctx, denoiserGraphName, map[string]any{
		"hidden_states":         latents,
		"encoder_hidden_states": promptEmbeds,
		"timestep":              timestep,
	}, nil)

	if err != nil {
		return nil, err
	}

	raw, ok := result.Outputs["sample"].([]float32)

	if !ok {
		return nil, fmt.Errorf("diffusion diagnose: denoiser output sample is %T", result.Outputs["sample"])
	}

	velocity := make([]float32, len(raw))
	copy(velocity, raw)

	return velocity, nil
}

func (harness *Harness) printLatentNormTrace(
	ctx context.Context,
	writer io.Writer,
	latentBaseline tensor.Tensor,
	promptEmbeds tensor.Tensor,
	timesteps []float32,
) error {
	fmt.Fprintf(writer, "\n=== latent L2 norm trace (4 denoiser forwards, host round-trips) ===\n")

	working, err := CloneTensor(harness.StateMemory, latentBaseline)

	if err != nil {
		return err
	}

	defer working.Close()

	for stepIndex, timestep := range timesteps {
		latentL2, err := TensorFloat32L2Norm(harness.StateMemory, working)

		if err != nil {
			return err
		}

		delta := harness.Scheduler.DeltaForStepIndex(stepIndex)

		fmt.Fprintf(
			writer,
			"step=%d timestep=%g latent_L2=%.6g delta=%g\n",
			stepIndex,
			timestep,
			latentL2,
			delta,
		)

		velocity, err := harness.denoiseVelocity(ctx, working, promptEmbeds, timestep)

		if err != nil {
			return err
		}

		latentValues, err := TensorToFloat32(harness.StateMemory, working)

		if err != nil {
			return err
		}

		for index := range latentValues {
			latentValues[index] += delta * velocity[index]
		}

		updated, err := uploadLatentValues(harness.StateMemory, working, latentValues)

		if err != nil {
			return err
		}

		working.Close()
		working = updated
	}

	return nil
}

func uploadLatentValues(
	memory tensor.Backend,
	template tensor.Tensor,
	values []float32,
) (tensor.Tensor, error) {
	storageDType, _, err := memory.Download(template)

	if err != nil {
		return nil, err
	}

	encoded := runtime.Float32AsDTypeBytes(values, storageDType)

	return memory.Upload(template.Shape(), storageDType, encoded)
}
