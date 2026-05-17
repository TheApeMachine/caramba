package chat

import (
	"context"
	"fmt"
	"io"
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
Generate satisfies the Generator interface. It runs the runtime
program once with the given prompt fed in through stdin and every
io.emit_text call routed back through the emit callback. This is the
one-shot path used by Session.RunPrompt and by callers that want a
single prompt → completion round-trip without taking over stdin.

For interactive multi-turn chat, Session.Run prefers the
SessionRunner.RunSession path below, which hands the terminal
streams straight to the executor so the manifest owns the entire
turn loop (state.history, state.kv, and friends accumulate naturally
across iterations of that outer loop).
*/
func (generator *RuntimeGenerator) Generate(
	ctx context.Context, prompt string, emit func(string) error,
) error {
	if emit == nil {
		return fmt.Errorf("chat: emit callback is required")
	}

	stdin := strings.NewReader(prompt + "\n")
	writer := &emitWriter{emit: emit}

	exec, err := generator.newExecutor(stdin, writer)

	if err != nil {
		return err
	}

	if err := exec.Run(ctx); err != nil {
		return err
	}

	return writer.err
}

/*
RunSession implements SessionRunner. The runtime manifest is
responsible for the outer turn loop, prompt printing, and command
handling; this method just constructs an executor wired to the
session's terminal streams and runs the program once. Persistent
state (history, KV cache, decode stream) is owned by the executor's
state objects and lives for the whole call, so context accumulates
across every turn the manifest reads.
*/
func (generator *RuntimeGenerator) RunSession(
	ctx context.Context, input io.Reader, output io.Writer,
) error {
	exec, err := generator.newExecutor(input, output)

	if err != nil {
		return err
	}

	return exec.Run(ctx)
}

func (generator *RuntimeGenerator) newExecutor(
	stdin io.Reader, stdout io.Writer,
) (*executor.Executor, error) {
	return executor.New(executor.Options{
		Program:       generator.program,
		Tokenizers:    map[string]tokenizer.Tokenizer{generator.tokenizerName: generator.tokenizer},
		GraphRunner:   generator.graphRunner,
		SamplerRunner: generator.samplerRunner,
		Telemetry:     generator.telemetry,
		Stdin:         stdin,
		Stdout:        stdout,
	})
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
