package chat

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/executor"
	"github.com/theapemachine/caramba/pkg/runtime/op"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/sampler"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
RuntimeGenerator is the manifest-driven chat generator. It replaces
the hand-coded decode loop in ModelGenerator with an Executor over a
runtime Program. Prompt formatting, tokenization, sampling, KV cache,
RoPE position state, and graph execution are all runtime operations
declared by the manifest rather than Go code paths.
*/
type RuntimeGenerator struct {
	program       *program.Program
	tokenizer     tokenizer.Tokenizer
	tokenizerName string
	graphRunner   op.GraphRunner
	samplerRunner op.SamplerRunner
	telemetry     telemetry.Recorder
}

/*
RuntimeGeneratorOptions configures a RuntimeGenerator. Program is the
compiled runtime IR; if nil, RuntimeChatTemplate is compiled. The
GraphRunner is required because chat's central activity is calling a
forward graph. TokenizerAssetID matches the asset name the manifest
references; it defaults to "tokenizer" because that is what the
canonical template uses.
*/
type RuntimeGeneratorOptions struct {
	Program          *program.Program
	Tokenizer        tokenizer.Tokenizer
	TokenizerAssetID string
	GraphRunner      op.GraphRunner
	SamplerRunner    op.SamplerRunner
	SamplerSeed      uint64
	ProjectRoot      string
	ManifestPath     string
	Telemetry        telemetry.Recorder
}

/*
NewRuntimeGenerator constructs a RuntimeGenerator from explicit
options. The default project root is "." and the default manifest is
RuntimeChatTemplate.
*/
func NewRuntimeGenerator(options RuntimeGeneratorOptions) (*RuntimeGenerator, error) {
	if options.Tokenizer == nil {
		return nil, fmt.Errorf("chat: tokenizer is required")
	}

	if options.GraphRunner == nil {
		return nil, fmt.Errorf("chat: graph runner is required")
	}

	runtimeProgram, err := resolveProgram(options)

	if err != nil {
		return nil, err
	}

	samplerRunner := options.SamplerRunner

	if samplerRunner == nil {
		samplerRunner = sampler.New(options.SamplerSeed)
	}

	tokenizerAsset := options.TokenizerAssetID

	if tokenizerAsset == "" {
		tokenizerAsset = "tokenizer"
	}

	return &RuntimeGenerator{
		program:       runtimeProgram,
		tokenizer:     options.Tokenizer,
		tokenizerName: tokenizerAsset,
		graphRunner:   options.GraphRunner,
		samplerRunner: samplerRunner,
		telemetry:     options.Telemetry,
	}, nil
}

/*
Generate satisfies the Generator interface. The prompt is fed into
the runtime's stdin so the program's io.read_line op picks it up;
each io.emit_token call inside the loop arrives here as a chunk and
is forwarded to the emit callback.
*/
func (generator *RuntimeGenerator) Generate(
	ctx context.Context, prompt string, emit func(string) error,
) error {
	if emit == nil {
		return fmt.Errorf("chat: emit callback is required")
	}

	stdin := strings.NewReader(prompt + "\n")
	writer := &emitWriter{emit: emit}

	exec, err := executor.New(executor.Options{
		Program:       generator.program,
		Tokenizers:    map[string]tokenizer.Tokenizer{generator.tokenizerName: generator.tokenizer},
		GraphRunner:   generator.graphRunner,
		SamplerRunner: generator.samplerRunner,
		Telemetry:     generator.telemetry,
		Stdin:         stdin,
		Stdout:        writer,
	})

	if err != nil {
		return err
	}

	if err := exec.Run(ctx); err != nil {
		return err
	}

	return writer.err
}

func resolveProgram(options RuntimeGeneratorOptions) (*program.Program, error) {
	if options.Program != nil {
		return options.Program, nil
	}

	root := options.ProjectRoot

	if root == "" {
		root = "."
	}

	runtimeCompiler := compiler.New(root)

	if options.ManifestPath != "" {
		return runtimeCompiler.Compile(options.ManifestPath)
	}

	data, err := asset.ReadFile(defaultChatRuntimeAsset)

	if err != nil {
		return nil, fmt.Errorf("chat: load default runtime manifest: %w", err)
	}

	return runtimeCompiler.CompileBytes(data)
}

const defaultChatRuntimeAsset = "runtime/chat.yml"

/*
emitWriter is the io.Writer adapter that turns each runtime stdout
write into an emit() callback invocation. It captures any error from
emit so the generator can surface it after Run returns.
*/
type emitWriter struct {
	emit func(string) error
	err  error
}

func (writer *emitWriter) Write(payload []byte) (int, error) {
	if writer.err != nil {
		return 0, writer.err
	}

	if err := writer.emit(string(payload)); err != nil {
		writer.err = err

		return 0, err
	}

	return len(payload), nil
}
