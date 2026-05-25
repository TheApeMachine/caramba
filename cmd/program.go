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
