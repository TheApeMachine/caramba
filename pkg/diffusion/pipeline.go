package diffusion

import (
	"context"
	"fmt"
	"math/rand"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/qpool"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

type Pipeline struct {
	config             Config
	backend            *compute.Backend
	tokenizer          tokenizer.Tokenizer
	textEncoderGraph   *manifest.Graph
	transformerGraph   *manifest.Graph
	textEncoderWeights *modelweights.Store
	transformerWeights *modelweights.Store
	telemetry          *Telemetry
}

type Result struct {
	Output string
	Width  int
	Height int
}

func NewPipeline(ctx context.Context, config Config) (*Pipeline, error) {
	telemetry := NewTelemetry(config)
	telemetry.Publish("manifest.resolve", "resolving diffusion manifest runtime")

	config, _, manifestPath, err := ResolveManifestConfig(config)

	if err != nil {
		return nil, telemetry.Error("manifest.resolve", "failed to resolve diffusion manifest", err)
	}

	telemetry.Apply(config)
	telemetry.Publish("manifest.resolved", "resolved diffusion manifest runtime")

	if err := config.ValidateRuntime(); err != nil {
		return nil, telemetry.Error("runtime.validate", "failed to validate diffusion runtime", err)
	}

	telemetry.Publish("backend.select", "selecting compute backend")

	backend, err := config.ComputeBackend()

	if err != nil {
		return nil, telemetry.Error("backend.select", "failed to select compute backend", err)
	}

	telemetry.SetBackend(string(backend.Location()))
	telemetry.Publish("backend.selected", "selected compute backend")
	telemetry.Publish("manifest.compile", "compiling denoiser manifest")

	transformerGraph, compiledManifest, err := CompileManifest(manifestPath)

	if err != nil {
		return nil, telemetry.Error("manifest.compile", "failed to compile denoiser manifest", err)
	}

	telemetry.Publish(
		"manifest.compiled",
		"compiled denoiser manifest",
		qpool.Field{Key: "compiled_manifest", Value: compiledManifest},
		qpool.Field{Key: "nodes", Value: len(transformerGraph.Nodes())},
	)
	telemetry.Publish("text_encoder.compile", "compiling prompt encoder manifest")

	textEncoderManifest := strings.TrimSpace(config.TextEncoder.Manifest)

	if textEncoderManifest == "" {
		return nil, telemetry.Error(
			"text_encoder.compile",
			"failed to compile prompt encoder manifest",
			fmt.Errorf("diffusion: runtime.text_encoder.manifest is required"),
		)
	}

	textEncoderGraph, _, err := CompileManifest(textEncoderManifest)

	if err != nil {
		return nil, telemetry.Error("text_encoder.compile", "failed to compile prompt encoder manifest", err)
	}

	telemetry.Publish(
		"text_encoder.compiled",
		"compiled prompt encoder manifest",
		qpool.Field{Key: "nodes", Value: len(textEncoderGraph.Nodes())},
	)
	telemetry.Publish("tokenizer.load", "loading diffusion tokenizer")

	tokenizerArtifact, err := tokenizer.Load(ctx, tokenizerSource(config.Tokenizer))

	if err != nil {
		return nil, telemetry.Error("tokenizer.load", "failed to load diffusion tokenizer", err)
	}

	telemetry.Publish(
		"tokenizer.loaded",
		"loaded diffusion tokenizer",
		qpool.Field{Key: "tokenizer_path", Value: tokenizerArtifact.Path},
		qpool.Field{Key: "tokenizer_backend", Value: tokenizerArtifact.Backend},
	)
	telemetry.Publish("weights.text_encoder", "resolving prompt encoder weights")

	textEncoderWeights, err := modelweights.Resolve(ctx, weightSource(config.TextEncoder))

	if err != nil {
		return nil, telemetry.Error("weights.text_encoder", "failed to resolve prompt encoder weights", err)
	}

	telemetry.Publish(
		"weights.text_encoder.resolved",
		"resolved prompt encoder weights",
		qpool.Field{Key: "tensors", Value: len(textEncoderWeights.Names())},
	)
	telemetry.Publish("weights.transformer", "resolving denoiser weights")

	transformerWeights, err := modelweights.Resolve(ctx, weightSource(config.Transformer))

	if err != nil {
		return nil, telemetry.Error("weights.transformer", "failed to resolve denoiser weights", err)
	}

	telemetry.Publish(
		"weights.transformer.resolved",
		"resolved denoiser weights",
		qpool.Field{Key: "tensors", Value: len(transformerWeights.Names())},
	)
	telemetry.Publish("runtime.ready", "diffusion runtime ready")

	return &Pipeline{
		config:             config,
		backend:            backend,
		tokenizer:          tokenizerArtifact.Tokenizer,
		textEncoderGraph:   textEncoderGraph,
		transformerGraph:   transformerGraph,
		textEncoderWeights: textEncoderWeights,
		transformerWeights: transformerWeights,
		telemetry:          telemetry,
	}, nil
}

func (pipeline *Pipeline) Close() error {
	if pipeline == nil || pipeline.backend == nil {
		return nil
	}

	return pipeline.backend.Close()
}

func (pipeline *Pipeline) Generate(ctx context.Context, prompt string) (Result, error) {
	if pipeline == nil || pipeline.backend == nil {
		return Result{}, fmt.Errorf("diffusion: pipeline is not initialized")
	}

	prompt = strings.TrimSpace(firstText(prompt, pipeline.config.Prompt))

	if prompt == "" {
		return Result{}, fmt.Errorf("diffusion: prompt is required")
	}

	imageSequenceLength, latentGridWidth, latentGridHeight, err := pipeline.latentLayout()

	if err != nil {
		return Result{}, err
	}

	pipeline.telemetry.Publish(
		"prompt.encode",
		"encoding prompt",
		qpool.Field{Key: "sequence_length", Value: imageSequenceLength},
	)

	promptEmbeds, err := pipeline.encodePrompt(ctx, prompt, imageSequenceLength)

	if err != nil {
		return Result{}, pipeline.telemetry.Error("prompt.encode", "failed to encode prompt", err)
	}

	latents := initialLatents(
		pipeline.config.Generation.Seed,
		imageSequenceLength*pipeline.config.Generation.LatentChannels,
	)
	scheduler, err := NewFlowMatchEulerScheduler(pipeline.config.Scheduler, imageSequenceLength)

	if err != nil {
		return Result{}, err
	}

	timesteps := scheduler.Timesteps()

	for stepIndex, timestep := range timesteps {
		pipeline.telemetry.Publish(
			"denoise.step",
			"running denoiser step",
			qpool.Field{Key: "step", Value: stepIndex + 1},
			qpool.Field{Key: "steps", Value: len(timesteps)},
			qpool.Field{Key: "timestep", Value: timestep},
		)

		modelOutput, err := pipeline.denoise(ctx, latents, promptEmbeds, timestep, imageSequenceLength)

		if err != nil {
			return Result{}, pipeline.telemetry.Error("denoise.step", "failed to run denoiser step", err)
		}

		latents, err = scheduler.Step(stepIndex, latents, modelOutput)

		if err != nil {
			return Result{}, err
		}
	}

	outputPath := pipeline.config.Generation.Output
	pipeline.telemetry.Publish("image.write", "writing latent preview image", qpool.Field{Key: "output", Value: outputPath})

	if err := WriteLatentPreview(outputPath, LatentImage{
		Width:    latentGridWidth,
		Height:   latentGridHeight,
		Channels: pipeline.config.Generation.LatentChannels,
		Values:   latents,
	}); err != nil {
		return Result{}, pipeline.telemetry.Error("image.write", "failed to write latent preview image", err)
	}

	return Result{
		Output: outputPath,
		Width:  latentGridWidth,
		Height: latentGridHeight,
	}, nil
}

func (pipeline *Pipeline) encodePrompt(
	ctx context.Context,
	prompt string,
	sequenceLength int,
) ([]float64, error) {
	formattedPrompt := applyPromptTemplate(pipeline.config.Generation.PromptTemplate, prompt)
	tokenIDs, err := pipeline.tokenizer.Encode(formattedPrompt)

	if err != nil {
		return nil, err
	}

	tokenIDs = fitTokenIDs(tokenIDs, sequenceLength, pipeline.config.Generation.PadTokenID)
	shape, err := tensor.NewShape([]int{1, sequenceLength})

	if err != nil {
		return nil, err
	}

	irGraph, err := manifest.LowerGraphToIR(pipeline.textEncoderGraph, shape)

	if err != nil {
		return nil, err
	}

	index, err := irGraph.Index()

	if err != nil {
		return nil, err
	}

	if err := bindInputValues(index, "input_ids", tokenValues(tokenIDs)); err != nil {
		return nil, err
	}

	if err := modelweights.BindIR(irGraph, pipeline.textEncoderWeights); err != nil {
		return nil, err
	}

	target := index.Node("prompt_embeds")

	if target == nil {
		return nil, fmt.Errorf("diffusion: prompt encoder target prompt_embeds is missing")
	}

	outputs, err := pipeline.backend.Execute(ctx, irGraph, []*ir.Node{target})

	if err != nil {
		return nil, err
	}

	defer closeOutputs(outputs)

	output := outputs[target.ID()]

	if output == nil {
		return nil, fmt.Errorf("diffusion: prompt encoder produced no output")
	}

	return output.CloneFloat64()
}

func (pipeline *Pipeline) denoise(
	ctx context.Context,
	latents []float64,
	promptEmbeds []float64,
	timestep float64,
	sequenceLength int,
) ([]float64, error) {
	latentShape, err := tensor.NewShape([]int{
		1,
		sequenceLength,
		pipeline.config.Generation.LatentChannels,
	})

	if err != nil {
		return nil, err
	}

	promptShape, err := tensor.NewShape([]int{1, sequenceLength, 7680})

	if err != nil {
		return nil, err
	}

	timestepShape, err := tensor.NewShape([]int{1})

	if err != nil {
		return nil, err
	}

	irGraph, err := manifest.LowerGraphToIRWithInputShapes(
		pipeline.transformerGraph,
		latentShape,
		map[string]tensor.Shape{
			"encoder_hidden_states": promptShape,
			"timestep":              timestepShape,
		},
	)

	if err != nil {
		return nil, err
	}

	index, err := irGraph.Index()

	if err != nil {
		return nil, err
	}

	if err := bindInputValues(index, "hidden_states", latents); err != nil {
		return nil, err
	}

	if err := bindInputValues(index, "encoder_hidden_states", promptEmbeds); err != nil {
		return nil, err
	}

	if err := bindInputValues(index, "timestep", []float64{timestep}); err != nil {
		return nil, err
	}

	if err := modelweights.BindIR(irGraph, pipeline.transformerWeights); err != nil {
		return nil, err
	}

	target := outputTarget(irGraph)

	if target == nil {
		return nil, fmt.Errorf("diffusion: denoiser has no executable target")
	}

	outputs, err := pipeline.backend.Execute(ctx, irGraph, []*ir.Node{target})

	if err != nil {
		return nil, err
	}

	defer closeOutputs(outputs)

	output := outputs[target.ID()]

	if output == nil {
		return nil, fmt.Errorf("diffusion: denoiser produced no output")
	}

	return output.CloneFloat64()
}

func (pipeline *Pipeline) latentLayout() (sequenceLength int, width int, height int, err error) {
	downsample := pipeline.config.Generation.LatentDownsample

	if downsample <= 0 {
		return 0, 0, 0, fmt.Errorf("diffusion: latent_downsample must be positive")
	}

	if pipeline.config.Generation.Width%downsample != 0 ||
		pipeline.config.Generation.Height%downsample != 0 {
		return 0, 0, 0, fmt.Errorf(
			"diffusion: image size %dx%d must be divisible by latent_downsample %d",
			pipeline.config.Generation.Width,
			pipeline.config.Generation.Height,
			downsample,
		)
	}

	width = pipeline.config.Generation.Width / downsample
	height = pipeline.config.Generation.Height / downsample
	sequenceLength = width * height

	return sequenceLength, width, height, nil
}

func tokenizerSource(source Source) tokenizer.Source {
	return tokenizer.Source{
		Source:   source.Source,
		File:     source.File,
		Cache:    source.Cache,
		Revision: source.Revision,
		RepoType: source.RepoType,
	}
}

func weightSource(source Source) modelweights.Source {
	return modelweights.Source{
		Source:   source.Source,
		File:     source.File,
		Cache:    source.Cache,
		Revision: source.Revision,
		RepoType: source.RepoType,
	}
}

func bindInputValues(index *ir.Index, inputID string, values []float64) error {
	inputNode := index.Node(inputID)

	if inputNode == nil {
		return fmt.Errorf("diffusion: manifest input %q is required", inputID)
	}

	if inputNode.OpType() != ir.OpInput {
		return fmt.Errorf("diffusion: manifest input %q is not an IR input node", inputID)
	}

	if inputNode.Shape().Len() != len(values) {
		return fmt.Errorf(
			"diffusion: input %q length %d does not match shape length %d",
			inputID,
			len(values),
			inputNode.Shape().Len(),
		)
	}

	inputNode.SetMetadata("values", values)

	return nil
}

func outputTarget(graph *ir.Graph) *ir.Node {
	index, err := graph.Index()

	if err != nil {
		return nil
	}

	if node := index.Node("proj_out"); node != nil {
		return node
	}

	sinks := graph.Sinks()

	for _, sink := range sinks {
		if sink.OpType() == ir.OpInput {
			continue
		}

		return sink
	}

	return nil
}

func closeOutputs(outputs map[string]tensor.Float64Tensor) {
	for _, output := range outputs {
		if output != nil {
			_ = output.Close()
		}
	}
}

func fitTokenIDs(tokenIDs []int, length int, padTokenID int) []int {
	fitted := make([]int, length)
	copy(fitted, tokenIDs)

	if len(tokenIDs) >= length {
		return fitted
	}

	for index := len(tokenIDs); index < len(fitted); index++ {
		fitted[index] = padTokenID
	}

	return fitted
}

func tokenValues(tokenIDs []int) []float64 {
	values := make([]float64, len(tokenIDs))

	for index, tokenID := range tokenIDs {
		values[index] = float64(tokenID)
	}

	return values
}

func initialLatents(seed int64, length int) []float64 {
	source := rand.New(rand.NewSource(seed))
	values := make([]float64, length)

	for index := range values {
		values[index] = source.NormFloat64()
	}

	return values
}

func applyPromptTemplate(template string, prompt string) string {
	if strings.TrimSpace(template) == "" {
		return prompt
	}

	return strings.ReplaceAll(template, "{{prompt}}", prompt)
}

func firstText(values ...string) string {
	for _, value := range values {
		text := strings.TrimSpace(value)

		if text != "" {
			return text
		}
	}

	return ""
}
