package diffusion

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/theapemachine/manifesto/tensor"
)

const diagnoseProgramPath = "runtime/diffusion-diagnose.yml"

/*
RunDiagnostics prints conditioning isolation results using Options (default: quick, 3 denoiser forwards).
*/
func RunDiagnostics(ctx context.Context, prompt string, writer io.Writer) error {
	return RunDiagnosticsWithOptions(ctx, prompt, writer, DefaultOptions())
}

/*
RunDiagnosticsWithOptions runs diagnostics with an explicit forward budget.
*/
func RunDiagnosticsWithOptions(
	ctx context.Context,
	prompt string,
	writer io.Writer,
	options Options,
) error {
	if writer == nil {
		writer = os.Stdout
	}

	harness, err := NewHarness(ctx, prompt)

	if err != nil {
		return err
	}

	defer harness.Close()

	latentBaseline, promptBaseline, err := harness.baselineTensors()

	if err != nil {
		return err
	}

	defer latentBaseline.Close()
	defer promptBaseline.Close()

	if err := harness.printPromptEmbedStats(writer, promptBaseline); err != nil {
		return err
	}

	timesteps := harness.Scheduler.Timesteps()

	if len(timesteps) == 0 {
		return fmt.Errorf("diffusion diagnose: scheduler produced no timesteps")
	}

	firstTimestep := timesteps[0]
	lastTimestep := timesteps[len(timesteps)-1]

	firstVelocity, err := harness.denoiseVelocity(ctx, latentBaseline, promptBaseline, firstTimestep)

	if err != nil {
		return err
	}

	lastVelocity, err := harness.denoiseVelocity(ctx, latentBaseline, promptBaseline, lastTimestep)

	if err != nil {
		return err
	}

	if err := harness.printTimestepAblationFromVelocities(
		writer,
		firstTimestep,
		lastTimestep,
		firstVelocity,
		lastVelocity,
	); err != nil {
		return err
	}

	if err := harness.printPromptAblationFromVelocity(
		ctx,
		writer,
		latentBaseline,
		promptBaseline,
		firstTimestep,
		firstVelocity,
	); err != nil {
		return err
	}

	if options.IncludeNormTrace {
		if err := harness.printLatentNormTrace(ctx, writer, latentBaseline, promptBaseline, timesteps); err != nil {
			return err
		}
	}

	return harness.printStepZeroVelocityFromSlice(writer, firstTimestep, firstVelocity)
}

func (harness *Harness) printPromptEmbedStats(writer io.Writer, promptEmbeds tensor.Tensor) error {
	l2, err := TensorFloat32L2Norm(harness.StateMemory, promptEmbeds)

	if err != nil {
		return err
	}

	shape := promptEmbeds.Shape()

	fmt.Fprintf(writer, "\n=== prompt_embeds after text encoder ===\n")
	fmt.Fprintf(writer, "shape=%v L2=%.6g\n", shape.Dims(), l2)

	if l2 < 1e-6 {
		fmt.Fprintf(writer, "VERDICT: prompt_embeds are zero — text encoder path broken\n")
	}

	return nil
}

func (harness *Harness) printTimestepAblationFromVelocities(
	writer io.Writer,
	firstTimestep float32,
	lastTimestep float32,
	firstVelocity []float32,
	lastVelocity []float32,
) error {
	l2, maxAbs, err := CompareVectors(firstVelocity, lastVelocity)

	if err != nil {
		return err
	}

	firstStats := StatsVector(firstVelocity)
	lastStats := StatsVector(lastVelocity)

	fmt.Fprintf(writer, "\n=== timestep ablation (2 denoiser forwards) ===\n")
	fmt.Fprintf(writer, "t_first=%g  t_last=%g\n", firstTimestep, lastTimestep)
	fmt.Fprintf(writer, "velocity L2(first,last)=%g  max_abs=%g\n", l2, maxAbs)
	fmt.Fprintf(writer, "first: norm=%.6g mean=%.6g std=%.6g\n", firstStats.Length, firstStats.Mean, firstStats.Std)
	fmt.Fprintf(writer, "last:  norm=%.6g mean=%.6g std=%.6g\n", lastStats.Length, lastStats.Mean, lastStats.Std)

	if l2 < 1e-3 {
		fmt.Fprintf(writer, "VERDICT: timestep path likely DEAD (velocities nearly identical)\n")
		return nil
	}

	fmt.Fprintf(writer, "VERDICT: timestep modulates denoiser output\n")

	return nil
}

func (harness *Harness) printPromptAblationFromVelocity(
	ctx context.Context,
	writer io.Writer,
	latents tensor.Tensor,
	promptEmbeds tensor.Tensor,
	timestep float32,
	realVelocity []float32,
) error {
	zeroPrompt, err := harness.zeroPromptTensor(promptEmbeds)

	if err != nil {
		return err
	}

	defer zeroPrompt.Close()

	garbageVelocity, err := harness.denoiseVelocity(ctx, latents, zeroPrompt, timestep)

	if err != nil {
		return err
	}

	l2, maxAbs, err := CompareVectors(realVelocity, garbageVelocity)

	if err != nil {
		return err
	}

	fmt.Fprintf(writer, "\n=== prompt ablation (1 denoiser forward, zeroed prompt) ===\n")
	fmt.Fprintf(writer, "timestep=%g\n", timestep)
	fmt.Fprintf(writer, "velocity L2(real,zero)=%g  max_abs=%g\n", l2, maxAbs)

	if l2 < 1e-3 {
		fmt.Fprintf(writer, "VERDICT: text conditioning likely DEAD (outputs nearly identical)\n")
		return nil
	}

	fmt.Fprintf(writer, "VERDICT: text conditioning changes denoiser output\n")

	return nil
}

func (harness *Harness) zeroPromptTensor(template tensor.Tensor) (tensor.Tensor, error) {
	if harness.zeroPrompt != nil {
		return CloneTensor(harness.StateMemory, harness.zeroPrompt)
	}

	storageDType, raw, err := harness.StateMemory.Download(template)

	if err != nil {
		return nil, err
	}

	zeroed := make([]byte, len(raw))

	zeroTensor, err := harness.StateMemory.Upload(template.Shape(), storageDType, zeroed)

	if err != nil {
		return nil, err
	}

	harness.zeroPrompt = zeroTensor

	return CloneTensor(harness.StateMemory, harness.zeroPrompt)
}

func (harness *Harness) printStepZeroVelocityFromSlice(
	writer io.Writer,
	timestep float32,
	velocity []float32,
) error {
	stats := StatsVector(velocity)
	previewCount := 8

	if previewCount > len(velocity) {
		previewCount = len(velocity)
	}

	fmt.Fprintf(writer, "\n=== step-0 velocity (from first forward, no extra pass) ===\n")
	fmt.Fprintf(writer, "timestep=%g len=%d\n", timestep, len(velocity))
	fmt.Fprintf(writer, "norm=%.6g mean=%.6g std=%.6g min=%g max=%g\n",
		stats.Length, stats.Mean, stats.Std, stats.Min, stats.Max)
	fmt.Fprintf(writer, "head[0:%d]:", previewCount)

	for index := range previewCount {
		fmt.Fprintf(writer, " %.6g", velocity[index])
	}

	fmt.Fprintf(writer, "\n")

	return nil
}
