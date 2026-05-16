package chat

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/qpool"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

const DefaultModelManifest = "model/llm/gpt2.yml"

/*
ModelConfig configures the manifest-backed chat generator.
*/
type ModelConfig struct {
	Runtime           string
	Backend           string
	Model             string
	Tokenizer         string
	TokenizerFile     string
	Manifest          string
	Cache             string
	Revision          string
	RepoType          string
	ModelCache        string
	ModelRevision     string
	ModelRepoType     string
	TokenizerCache    string
	TokenizerRevision string
	TokenizerRepoType string
	MaxNewTokens      int
	RepetitionPenalty float64
	Temperature       float64
	TopK              int
	TopP              float64
	Seed              int64
	StopTokens        []string
	StopSpecialTokens bool
}

/*
ModelGenerator lowers model manifests into compute backend execution.
*/
type ModelGenerator struct {
	backend   *compute.Backend
	graph     *manifest.Graph
	tokenizer tokenizer.Tokenizer
	weights   *modelweights.Store
	cache     *kv.Cache
	manifest  string
	model     string
	maxTokens int
	policy    generationPolicy
}

/*
NewModelGenerator validates model chat configuration.
*/
func NewModelGenerator(ctx context.Context, config ModelConfig) (*ModelGenerator, error) {
	telemetry := newModelStartupTelemetry(config)
	telemetry.Publish("model.start", "starting manifest-backed model runtime")
	telemetry.Publish("manifest.resolve", "resolving model manifest runtime")

	config, err := resolveManifestModelConfig(config)

	if err != nil {
		return nil, telemetry.Error(
			"manifest.resolve",
			"failed to resolve model manifest runtime",
			err,
		)
	}

	telemetry.Apply(config)
	telemetry.Publish(
		"manifest.resolved",
		"resolved model manifest runtime",
		qpool.Field{Key: "max_new_tokens", Value: config.MaxNewTokens},
		qpool.Field{Key: "temperature", Value: config.Temperature},
		qpool.Field{Key: "top_k", Value: config.TopK},
		qpool.Field{Key: "top_p", Value: config.TopP},
	)

	if err := config.ValidateRuntime(); err != nil {
		return nil, telemetry.Error(
			"runtime.validate",
			"failed to validate model runtime",
			err,
		)
	}

	if config.MaxNewTokens < 1 {
		err := fmt.Errorf("chat.model: max_new_tokens must be positive")

		return nil, telemetry.Error(
			"generation.validate",
			"failed to validate generation policy",
			err,
		)
	}

	telemetry.Publish("generation.policy", "building generation policy")

	policy, err := newGenerationPolicy(config)

	if err != nil {
		return nil, telemetry.Error(
			"generation.policy",
			"failed to build generation policy",
			err,
		)
	}

	telemetry.Publish("backend.select", "selecting compute backend")

	backend, err := config.ComputeBackend()

	if err != nil {
		return nil, telemetry.Error(
			"backend.select",
			"failed to select compute backend",
			err,
		)
	}

	telemetry.SetBackend(string(backend.Location()))
	telemetry.Publish("backend.selected", "selected compute backend")
	telemetry.Publish("manifest.compile", "compiling model manifest")

	graph, manifestPath, err := compileModelManifest(config.Manifest)

	if err != nil {
		return nil, telemetry.Error(
			"manifest.compile",
			"failed to compile model manifest",
			err,
		)
	}

	telemetry.Publish(
		"manifest.compiled",
		"compiled model manifest",
		qpool.Field{Key: "compiled_manifest", Value: manifestPath},
		qpool.Field{Key: "nodes", Value: len(graph.Nodes())},
	)

	source := tokenizerSource(config)

	if source.Source == "" {
		err := fmt.Errorf("chat.model: model or tokenizer source is required")

		return nil, telemetry.Error(
			"tokenizer.source",
			"failed to resolve tokenizer source",
			err,
		)
	}

	telemetry.Publish(
		"tokenizer.load",
		"loading tokenizer",
		qpool.Field{Key: "tokenizer_source", Value: source.Source},
		qpool.Field{Key: "tokenizer_file", Value: source.WithDefaults().File},
		qpool.Field{Key: "tokenizer_revision", Value: source.Revision},
	)

	artifact, err := tokenizer.Load(ctx, source)

	if err != nil {
		return nil, telemetry.Error(
			"tokenizer.load",
			"failed to load tokenizer",
			err,
			qpool.Field{Key: "tokenizer_source", Value: source.Source},
		)
	}

	telemetry.Publish(
		"tokenizer.loaded",
		"loaded tokenizer",
		qpool.Field{Key: "tokenizer_path", Value: artifact.Path},
		qpool.Field{Key: "tokenizer_backend", Value: artifact.Backend},
	)

	if err := policy.bindStopSequences(artifact.Tokenizer); err != nil {
		return nil, telemetry.Error(
			"generation.stop_sequences",
			"failed to bind generation stop sequences",
			err,
		)
	}

	weightSource := modelWeightSource(config)
	telemetry.Publish(
		"weights.resolve",
		"resolving model weights",
		qpool.Field{Key: "weight_source", Value: weightSource.Source},
		qpool.Field{Key: "weight_revision", Value: weightSource.Revision},
	)

	weightStore, err := modelweights.Resolve(ctx, weightSource)

	if err != nil {
		return nil, telemetry.Error(
			"weights.resolve",
			"failed to resolve model weights",
			err,
			qpool.Field{Key: "weight_source", Value: weightSource.Source},
		)
	}

	telemetry.Publish(
		"weights.resolved",
		"resolved model weights",
		qpool.Field{Key: "tensors", Value: len(weightStore.Names())},
	)
	telemetry.Publish("runtime.ready", "model runtime ready")

	return &ModelGenerator{
		backend:   backend,
		graph:     graph,
		tokenizer: artifact.Tokenizer,
		weights:   weightStore,
		cache:     kv.NewCache(),
		manifest:  manifestPath,
		model:     config.Model,
		maxTokens: config.MaxNewTokens,
		policy:    policy,
	}, nil
}

/*
ComputeBackend resolves the model runtime backend.
An empty value keeps package callers on CPU when no manifest runtime is present.
*/
func (config ModelConfig) ComputeBackend() (*compute.Backend, error) {
	backendName := strings.ToLower(strings.TrimSpace(config.Backend))

	if backendName == "" {
		backendName = "cpu"
	}

	backendType, err := modelBackendType(backendName)

	if err != nil {
		return nil, err
	}

	return compute.NewBackend(backendType), nil
}

/*
ValidateRuntime rejects manifest runtime types that are not model execution.
*/
func (config ModelConfig) ValidateRuntime() error {
	runtimeName := strings.ToLower(strings.TrimSpace(config.Runtime))

	if runtimeName == "" || runtimeName == "model" {
		return nil
	}

	return fmt.Errorf("chat.model: unsupported manifest runtime %q", config.Runtime)
}

/*
BackendName reports the concrete compute location used by this generator.
*/
func (generator *ModelGenerator) BackendName() string {
	if generator == nil || generator.backend == nil {
		return ""
	}

	return string(generator.backend.Location())
}

/*
ModelName reports the serialized model source configured by the manifest.
*/
func (generator *ModelGenerator) ModelName() string {
	if generator == nil {
		return ""
	}

	return generator.model
}

func modelBackendType(backendName string) (compute.BackendType, error) {
	switch backendName {
	case "auto":
		return automaticModelBackendType(), nil
	case "cpu", "host":
		return compute.CPU, nil
	case "metal":
		return compute.METAL, nil
	case "cuda":
		return compute.CUDA, nil
	case "xla":
		return compute.XLA, nil
	default:
		return compute.CPU, fmt.Errorf("chat.model: unsupported backend %q", backendName)
	}
}

func automaticModelBackendType() compute.BackendType {
	switch runtime.GOOS {
	case "darwin":
		return compute.METAL
	case "linux":
		return compute.CUDA
	default:
		return compute.CPU
	}
}

/*
Generate streams model tokens.
*/
func (generator *ModelGenerator) Generate(
	ctx context.Context,
	prompt string,
	emit func(string) error,
) error {
	if generator == nil || generator.graph == nil ||
		generator.tokenizer == nil || generator.weights == nil || generator.cache == nil {
		return fmt.Errorf("chat.model: generator is not initialized")
	}

	tokenIDs, err := generator.tokenizer.Encode(prompt)

	if err != nil {
		return err
	}

	generator.cache.Reset()

	if err := generator.cache.SetCapacity(len(tokenIDs) + generator.maxTokens); err != nil {
		return err
	}

	historyTokenIDs := append([]int(nil), tokenIDs...)
	inputTokenIDs := tokenIDs
	positionStart := 0
	generatedTokenIDs := make([]int, 0, generator.maxTokens)
	pendingTokenIDs := make([]int, 0)
	stream := newTokenStream(generator.tokenizer)

	for range generator.maxTokens {
		nextTokenID, err := generator.nextToken(
			ctx,
			inputTokenIDs,
			positionStart,
			historyTokenIDs,
		)

		if err != nil {
			return err
		}

		historyTokenIDs = append(historyTokenIDs, nextTokenID)
		generatedTokenIDs = append(generatedTokenIDs, nextTokenID)
		pendingTokenIDs = append(pendingTokenIDs, nextTokenID)
		inputTokenIDs = []int{nextTokenID}
		positionStart = len(historyTokenIDs) - 1

		if generator.policy.stopMatched(generatedTokenIDs) {
			return nil
		}

		if generator.policy.stopPending(generatedTokenIDs) {
			continue
		}

		if err := stream.Append(pendingTokenIDs, emit); err != nil {
			return err
		}

		pendingTokenIDs = pendingTokenIDs[:0]
	}

	if err := stream.Append(pendingTokenIDs, emit); err != nil {
		return err
	}

	return stream.Flush(emit)
}

func (generator *ModelGenerator) nextToken(
	ctx context.Context,
	inputTokenIDs []int,
	positionStart int,
	historyTokenIDs []int,
) (int, error) {
	irGraph, err := generator.lower(inputTokenIDs, positionStart)

	if err != nil {
		return 0, err
	}

	if err := modelweights.BindIR(irGraph, generator.weights); err != nil {
		return 0, err
	}

	target := outputTarget(irGraph)

	if target == nil {
		return 0, fmt.Errorf("chat.model: manifest %s has no executable targets", generator.manifest)
	}

	outputs, err := generator.backend.Execute(ctx, irGraph, []*ir.Node{target})

	if err != nil {
		return 0, fmt.Errorf(
			"chat.model: manifest %s lowered to %s backend IR (%d nodes) with serialized model %q: %w",
			generator.manifest,
			generator.backend.Location(),
			len(irGraph.Nodes()),
			generator.model,
			err,
		)
	}

	defer func() {
		for _, output := range outputs {
			if output != nil {
				_ = output.Close()
			}
		}
	}()

	output := outputs[target.ID()]

	if output == nil {
		return 0, fmt.Errorf("chat.model: target %q produced no output", target.ID())
	}

	values, err := output.CloneFloat64()

	if err != nil {
		return 0, err
	}

	nextTokenID, err := selectLastToken(
		output.Shape().Dims(),
		values,
		historyTokenIDs,
		generator.policy,
	)

	if err != nil {
		return 0, err
	}

	return nextTokenID, nil
}

func outputTarget(graph *ir.Graph) *ir.Node {
	index, err := graph.Index()

	if err != nil {
		return nil
	}

	if node := index.Node("lm_head"); node != nil {
		return node
	}

	sinks := graph.Sinks()

	if len(sinks) == 0 {
		return nil
	}

	return sinks[0]
}

func (generator *ModelGenerator) lower(tokenIDs []int, positionStart int) (*ir.Graph, error) {
	shape, err := tensor.NewShape([]int{1, len(tokenIDs)})

	if err != nil {
		return nil, err
	}

	irGraph, err := manifest.LowerGraphToIR(generator.graph, shape)

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

	generator.bindKVCache(irGraph)
	positionNode := index.Node("position_ids")

	if positionNode == nil {
		return irGraph, nil
	}

	positionNode.SetMetadata("values", positionValues(positionStart, len(tokenIDs)))

	return irGraph, nil
}

func (generator *ModelGenerator) bindKVCache(irGraph *ir.Graph) {
	if generator.cache == nil {
		return
	}

	for _, node := range irGraph.Nodes() {
		operationID := string(node.OperationID())

		if operationID != "attention.sdpa" && operationID != "attention.gqa" {
			continue
		}

		causal, _ := node.Metadata()["causal"].(bool)

		if !causal {
			continue
		}

		node.SetMetadata("kv_cache", generator.cache)
	}
}

func bindInputValues(index *ir.Index, inputID string, values []float64) error {
	inputNode := index.Node(inputID)

	if inputNode == nil {
		return fmt.Errorf("chat.model: manifest input %q is required", inputID)
	}

	if inputNode.OpType() != ir.OpInput {
		return fmt.Errorf("chat.model: manifest input %q is not an IR input node", inputID)
	}

	inputNode.SetMetadata("values", values)

	return nil
}

func tokenValues(tokenIDs []int) []float64 {
	values := make([]float64, len(tokenIDs))

	for index, tokenID := range tokenIDs {
		values[index] = float64(tokenID)
	}

	return values
}

func positionValues(start int, count int) []float64 {
	values := make([]float64, count)

	for index := range values {
		values[index] = float64(start + index)
	}

	return values
}

func tokenizerSource(config ModelConfig) tokenizer.Source {
	source := strings.TrimSpace(config.Tokenizer)

	if source == "" {
		source = strings.TrimSpace(config.Model)
	}

	if source == "" {
		return tokenizer.Source{}
	}

	return tokenizer.Source{
		Source:   source,
		File:     strings.TrimSpace(config.TokenizerFile),
		Cache:    firstText(config.TokenizerCache, config.Cache),
		Revision: firstText(config.TokenizerRevision, config.Revision),
		RepoType: firstText(config.TokenizerRepoType, config.RepoType),
	}
}

func modelWeightSource(config ModelConfig) modelweights.Source {
	source := strings.TrimSpace(config.Model)

	if source == "" {
		source = strings.TrimSpace(config.Tokenizer)
	}

	return modelweights.Source{
		Source:   source,
		Cache:    firstText(config.ModelCache, config.Cache),
		Revision: firstText(config.ModelRevision, config.Revision),
		RepoType: firstText(config.ModelRepoType, config.RepoType),
	}
}

func compileModelManifest(path string) (*manifest.Graph, string, error) {
	path = strings.TrimSpace(path)

	if path == "" {
		path = DefaultModelManifest
	}

	graph, err := compileLocalManifest(path)

	if err == nil {
		return graph, path, nil
	}

	if !errors.Is(err, os.ErrNotExist) {
		return nil, path, err
	}

	data, err := asset.ReadFile(path)

	if err != nil {
		return nil, path, fmt.Errorf("chat.model: manifest %s: %w", path, err)
	}

	graph, err = manifest.NewCompiler(".").CompileBytes(data)

	if err != nil {
		return nil, path, err
	}

	return graph, path, nil
}

func compileLocalManifest(path string) (*manifest.Graph, error) {
	info, err := os.Stat(path)

	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		return nil, fmt.Errorf("chat.model: manifest %s is a directory", path)
	}

	if filepath.IsAbs(path) {
		return manifest.NewCompiler(filepath.Dir(path)).Compile(filepath.Base(path))
	}

	return manifest.NewCompiler(".").Compile(path)
}
