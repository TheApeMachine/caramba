package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/config"
	weightstore "github.com/theapemachine/caramba/pkg/weights"
	"github.com/theapemachine/hf/hub"
	"github.com/theapemachine/hf/program"
	"github.com/theapemachine/hf/safetensors"
	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/typer"
	"github.com/theapemachine/manifesto/types"
	"github.com/theapemachine/puter/execution"
	"github.com/theapemachine/puter/pool"
)

// devicePoolPinTarget is the tensor.Location the chat / program path
// pins its device pool to. Today this is Host because the dispatcher
// still loads weights through tensor.Backend.Upload and the Metal
// backend's Upload returns a DeviceTensor whose MTLBuffer is not yet
// fully wired (puter ARCHITECTURE.md §3.1 lines 1223-1237 and
// GAPS.md §8.3 step 6 — "brings chat.yml up on CPU before validating
// the same chain against Metal"). Without pinning, MemoryBackend
// picks Metal, Upload returns a half-initialised DeviceTensor, and
// the first kernel dispatch crashes with SIGSEGV at offset 0x20
// (the buffer field of an uninitialised metal.DeviceTensor).
//
// Once the buffer-handle plumbing lands, flip this to tensor.Metal
// (or wire it to a CLI flag) — the rest of program.go below does not
// need to change.
var devicePoolPinTarget = tensor.Host

func runProgram(
	command *cobra.Command,
	programPath string,
	initialValues map[string]any,
) error {
	hubConfig := config.NewHubConfig()
	qpoolConfig := config.NewQPoolConfig()

	workerPool := qpoolConfig.NewWorkerPool(command.Context())

	defer workerPool.Close()

	devicePool, err := pool.New(command.Context(), workerPool)

	if err != nil {
		return fmt.Errorf("caramba: discover devices: %w", err)
	}

	defer devicePool.Close()

	// Narrow the discovered devices to devicePoolPinTarget. See the
	// constant's doc-comment for why. Errors here mean the target
	// location isn't present (e.g. asking for Metal on Linux); fall
	// through with the discovered pool so the diagnostic surfaces at
	// the first kernel dispatch rather than at pool init.
	_ = devicePool.PinTo(devicePoolPinTarget)

	stateMemory, _, err := devicePool.MemoryBackend()

	if err != nil {
		return fmt.Errorf("caramba: resolve state memory: %w", err)
	}

	graphBackend := execution.New(devicePool)
	defer graphBackend.Close()

	hubClient := hub.NewClient(hubConfig)
	hubAdapter := hub.NewResolveAdapter(hubClient)

	if err := attachWeights(
		command, programPath, hubClient, hubConfig, stateMemory, graphBackend,
	); err != nil {
		return err
	}

	includeResolver, err := program.NewIncludeResolver(program.IncludeResolverOptions{
		Hub:      hubAdapter,
		CacheDir: hubConfig.CacheDir,
	})

	if err != nil {
		return fmt.Errorf("caramba: build include resolver: %w", err)
	}

	programOrchestrator, err := runtime.NewOrchestrator(runtime.OrchestratorOptions{
		Hub: hubAdapter,
		Parser: func(archive []byte) (types.Parser, error) {
			return safetensors.NewParser(archive)
		},
		Compute:         graphBackend,
		Host:            program.NewHost(program.HostOptions{Stdin: os.Stdin, HubConfig: hubConfig}),
		StateMemory:     stateMemory,
		CacheDir:        hubConfig.CacheDir,
		Stdin:           os.Stdin,
		InitialValues:   initialValues,
		IncludeResolver: includeResolver,

		// The typer can synthesize `shape.cast` adaptor nodes when the
		// inferred edge dtype differs from a consumer's expected dtype.
		// Those nodes are inserted into the AST but the compiler does
		// not currently propagate the rewrite into the DAG that drives
		// execution planning, so the dispatcher would walk past them
		// and fail to find their named outputs. The dispatcher's input
		// handlers (e.g. valueTable.tokenIDs) already accept a range of
		// integer widths directly, so we disable adaptor synthesis and
		// let the handlers consume the original producer values.
		ConfigureTyper: true,
		TyperOptions:   typer.Options{DisableSynthesis: true},

		// The static memory planner needs concrete values for every
		// symbolic shape dimension before it can size the workspace.
		// Runtime symbols below are upper bounds the program will be
		// allowed to use; the activation buffers are pre-allocated
		// to the worst case and kernels touch only the live prefix.
		//
		// "N" is the typer's wildcard symbol used by ops like
		// embedding.token that take a "[N]" indices tensor — it lines
		// up with the chat program's max-sequence guarantee declared
		// by the KV cache page count (chat.yml's key_pages first dim
		// is 4096). "T" is the conventional sequence-length symbol
		// surface area uses for transformer activations.
		PlannerBindings: ir.SymbolMap{
			"N": 4096,
			"T": 4096,
			"B": 1,
		},
	})

	if err != nil {
		return err
	}

	return programOrchestrator.Run(command.Context(), programPath)
}

/*
attachWeights resolves every `hf://...` include declared by the program
manifest, downloads each repository's safetensors archives into the local
HF cache, and wires the resulting WeightStore into the graph backend.

Programs that declare no `hf://` includes (e.g. unit-test fixtures) are
allowed to run without weights — the nil fallback in execution.Backend
remains in place and any weighted node will surface ErrWeightNotFound at
dispatch time, which is the right failure mode for those programs.
*/
func attachWeights(
	command *cobra.Command,
	programPath string,
	hubClient *hub.Client,
	hubConfig *hub.HubConfig,
	memory tensor.Backend,
	graphBackend *execution.Backend,
) error {
	programYAML, err := asset.ReadFile(programPath)

	if err != nil {
		return fmt.Errorf("caramba: read program %q: %w", programPath, err)
	}

	refs, err := weightstore.ExtractHFReferences(programYAML)

	if err != nil {
		return fmt.Errorf("caramba: scan program includes: %w", err)
	}

	paths := make([]string, 0)

	for _, ref := range refs {
		downloaded, err := weightstore.DownloadSafetensors(
			command.Context(),
			hubClient,
			ref.RepoID,
			"",
			hubConfig.CacheDir,
			hubConfig.Token,
		)

		if err != nil {
			return fmt.Errorf("caramba: download weights for %q: %w", ref.RepoID, err)
		}

		paths = append(paths, downloaded...)
	}

	if len(paths) == 0 {
		return nil
	}

	store, err := weightstore.New(memory, paths)

	if err != nil {
		return fmt.Errorf("caramba: build weight store: %w", err)
	}

	graphBackend.WithWeights(store)

	return nil
}
