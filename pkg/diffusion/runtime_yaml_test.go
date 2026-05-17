package diffusion

import (
	"context"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/executor"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/scheduler"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
testDiffusionYAML mirrors the embedded diffusion.yml at sizes a unit
test can drive against stub graphs. It is intentionally inline so the
test exercises the compiler reading the same YAML shape the embedded
manifest uses.
*/
const testDiffusionYAML = `
variables:
  generation:
    sequence_length: 4
    latent_channels: 3
    prompt_embed_dim: 8
    output_width: 2
    output_height: 2
    output_channels: 3
    seed: 42
    steps: 2

system:
  runtime:
    type: program
    entry: image
    state:
      latents:
        type: tensor
        shape:
          - 1
          - ${generation.sequence_length}
          - ${generation.latent_channels}
        init: gaussian
        seed: ${generation.seed}
      step_index:
        type: counter
        initial: 0
    schedulers:
      scheduler:
        type: flow_match_euler_discrete
        steps: ${generation.steps}
        num_train_timesteps: 1000
    graphs:
      text_encoder:
        topology: text_encoder
      denoiser:
        topology: denoiser
      vae:
        topology: vae
    program:
      - id: read_prompt
        op: io.read_line
        outputs: { text: prompt }
      - id: encode_prompt
        op: tokenizer.encode
        tokenizer: tokenizer
        text: prompt
        outputs: { tokens: input_ids }
      - id: encode_text
        op: graph.call
        graph: text_encoder
        inputs: { input_ids: input_ids }
        outputs: { prompt_embeds: prompt_embeds }
      - id: compute_timesteps
        op: scheduler.timesteps
        scheduler: scheduler
        outputs: { timesteps: timesteps }
      - id: denoise_loop
        op: control.loop_each
        source: timesteps
        as: timestep
        body:
          - id: denoise
            op: graph.call
            graph: denoiser
            inputs:
              hidden_states: state.latents
              encoder_hidden_states: prompt_embeds
              timestep: timestep
            outputs:
              velocity: velocity
          - id: scheduler_step
            op: scheduler.step
            scheduler: scheduler
            step_index: state.step_index
            latents: state.latents
            velocity: velocity
            outputs:
              latents: state.latents
          - id: advance_step
            op: state.update
            update: increment
            target: state.step_index
      - id: vae_decode
        op: graph.call
        graph: vae
        inputs: { latents: state.latents }
        outputs: { image: image }
      - id: write_image
        op: io.write_image
        image: image
        config:
          width: 2
          height: 2
          channels: 3
          layout: channel_interleaved
          range: neg_one_one
`

type stubTokenizer struct{}

func (stubTokenizer) Encode(text string) ([]int, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}

	out := []int{}

	for _, field := range strings.Fields(text) {
		out = append(out, len(field))
	}

	return out, nil
}

func (stubTokenizer) Decode(ids []int, skip bool) (string, error)        { return "", nil }
func (stubTokenizer) VocabSize() int                                     { return 1024 }
func (stubTokenizer) SpecialTokenIDs() []int                             { return nil }

type stubDiffusionRunner struct {
	sequenceLength int
	latentChannels int
	embedDim       int
}

func (runner stubDiffusionRunner) Call(
	ctx context.Context, module program.GraphModule, inputs map[string]any,
) (map[string]any, error) {
	switch module.ID {
	case "text_encoder":
		size := runner.sequenceLength * runner.embedDim
		out := make([]float64, size)

		for index := range out {
			out[index] = 1
		}

		return map[string]any{"prompt_embeds": out}, nil
	case "denoiser":
		size := runner.sequenceLength * runner.latentChannels
		out := make([]float64, size)

		for index := range out {
			out[index] = 0.1
		}

		return map[string]any{"velocity": out}, nil
	case "vae":
		latents, ok := inputs["latents"].(*state.Tensor)

		if !ok {
			return nil, fmt.Errorf("stub vae: latents must be *state.Tensor")
		}

		values := latents.Values()
		out := make([]float64, len(values))

		for index, value := range values {
			out[index] = value * 0.5
		}

		return map[string]any{"image": out}, nil
	}

	return nil, fmt.Errorf("stub: unknown graph %q", module.ID)
}

func TestDiffusionRuntimeYAML(t *testing.T) {
	Convey("Given the diffusion runtime manifest compiled from YAML", t, func() {
		runtimeProgram, err := compiler.New(".").CompileBytes([]byte(testDiffusionYAML))
		So(err, ShouldBeNil)
		So(runtimeProgram.Validate(), ShouldBeNil)

		tmpDir := t.TempDir()
		outputPath := filepath.Join(tmpDir, "out.png")

		// Inject the resolved output path into the write_image step config —
		// the manifest leaves it variable so tests can redirect into TempDir.
		writeStep := runtimeProgram.FindStep("write_image")
		So(writeStep, ShouldNotBeNil)
		writeStep.Config["path"] = outputPath

		runner := stubDiffusionRunner{
			sequenceLength: 4,
			latentChannels: 3,
			embedDim:       8,
		}

		stdin := strings.NewReader("a cat on a mat\n")

		exec, err := executor.New(executor.Options{
			Program:         runtimeProgram,
			Tokenizers:      map[string]tokenizer.Tokenizer{"tokenizer": stubTokenizer{}},
			GraphRunner:     runner,
			SchedulerRunner: scheduler.NewFlowMatchEuler(),
			Stdin:           stdin,
		})
		So(err, ShouldBeNil)

		Convey("Running the program should drive denoise loop and write a PNG", func() {
			So(exec.Run(context.Background()), ShouldBeNil)

			info, err := os.Stat(outputPath)
			So(err, ShouldBeNil)
			So(info.Size(), ShouldBeGreaterThan, 0)

			file, err := os.Open(outputPath)
			So(err, ShouldBeNil)
			defer file.Close()

			img, err := png.Decode(file)
			So(err, ShouldBeNil)
			So(img.Bounds().Dx(), ShouldEqual, 2)
			So(img.Bounds().Dy(), ShouldEqual, 2)
		})

		Convey("Counter should increment once per timestep", func() {
			So(exec.Run(context.Background()), ShouldBeNil)
			counter := exec.States()["step_index"].(*state.Counter)
			So(counter.Value(), ShouldEqual, 2)
		})
	})
}
