package runtime

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/hf/hub"
	"github.com/theapemachine/hf/program"
	"github.com/theapemachine/hf/safetensors"
	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/compiler"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/parse"
	"github.com/theapemachine/manifesto/resolve"
	manifestoruntime "github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/types"
	"github.com/theapemachine/puter/execution"
	"github.com/theapemachine/puter/pool"
	"github.com/theapemachine/qpool"
)

/*
Platform wires puter device execution with manifesto orchestration and hf host IO.
*/
type Platform struct {
	devicePool   *pool.Pool
	compute      *execution.Backend
	weights      *execution.ResidentStore
	workerPool   *qpool.Q[any]
	weightParser types.Parser
}

/*
PlatformOptions configures one manifest-driven runtime session.
*/
type PlatformOptions struct {
	ComputeConfig *config.ComputeConfig
	HubConfig     *hub.HubConfig
	Stdin         io.Reader
}

/*
NewPlatform discovers devices and prepares the execution backend.
*/
func NewPlatform(ctx context.Context, options PlatformOptions) (*Platform, error) {
	if options.ComputeConfig == nil {
		options.ComputeConfig = config.NewComputeConfig()
	}

	if options.HubConfig == nil {
		options.HubConfig = config.NewHubConfig()
	}

	workerPool := config.NewQPoolConfig().NewWorkerPool(ctx)
	devicePool, err := pool.New(ctx, workerPool)

	if err != nil {
		return nil, fmt.Errorf("runtime platform: device pool: %w", err)
	}

	location, err := resolveComputeLocation(options.ComputeConfig.Device)

	if err != nil {
		_ = devicePool.Close()
		return nil, err
	}

	if err := devicePool.PinTo(location); err != nil {
		_ = devicePool.Close()
		return nil, fmt.Errorf("runtime platform: pin device %s: %w", location, err)
	}

	return &Platform{
		devicePool: devicePool,
		compute:    execution.New(devicePool),
		workerPool: workerPool,
	}, nil
}

/*
RunProgram compiles and executes one manifest program path end-to-end.
*/
func (platform *Platform) RunProgram(ctx context.Context, programPath string, options PlatformOptions) error {
	if platform == nil || platform.compute == nil || platform.devicePool == nil {
		return fmt.Errorf("runtime platform: not initialized")
	}

	if options.ComputeConfig == nil {
		options.ComputeConfig = config.NewComputeConfig()
	}

	if options.HubConfig == nil {
		options.HubConfig = config.NewHubConfig()
	}

	stdin := options.Stdin

	if stdin == nil {
		stdin = os.Stdin
	}

	hubClient := hub.NewClient(options.HubConfig)
	resolveHub := hub.NewResolveAdapter(hubClient)
	includeResolver, err := program.NewIncludeResolver(program.IncludeResolverOptions{
		Hub:      resolveHub,
		CacheDir: options.HubConfig.CacheDir,
		Token:    options.HubConfig.Token,
	})

	if err != nil {
		return fmt.Errorf("runtime platform: include resolver: %w", err)
	}

	memoryBackend, _, err := platform.devicePool.MemoryBackend()

	if err != nil {
		return fmt.Errorf("runtime platform: memory backend: %w", err)
	}

	weightParser, weightStore, err := loadProgramWeights(
		ctx,
		programPath,
		includeResolver,
		resolveHub,
		options.HubConfig,
		memoryBackend,
	)

	if err != nil {
		return fmt.Errorf("runtime platform: load weights: %w", err)
	}

	platform.weights = weightStore
	platform.weightParser = weightParser
	platform.compute = platform.compute.WithWeights(weightStore)

	host := program.NewHost(program.HostOptions{
		Stdin:     stdin,
		HubConfig: options.HubConfig,
	})

	orchestrator, err := manifestoruntime.NewOrchestrator(manifestoruntime.OrchestratorOptions{
		Hub:             resolveHub,
		Parser:          newSafetensorsParser,
		Compute:         platform.compute,
		Host:            host,
		StateMemory:     memoryBackend,
		CacheDir:        options.HubConfig.CacheDir,
		Stdin:           stdin,
		IncludeResolver: includeResolver,
		WeightParser:    weightParser,
		PlannerBindings: plannerBindingsForProgram(programPath),
	})

	if err != nil {
		return fmt.Errorf("runtime platform: orchestrator: %w", err)
	}

	return orchestrator.Run(ctx, programPath)
}

/*
Close releases devices, weights, and worker pool resources.
*/
func (platform *Platform) Close() error {
	if platform == nil {
		return nil
	}

	if platform.weights != nil {
		_ = platform.weights.Close()
		platform.weights = nil
	}

	if platform.compute != nil {
		_ = platform.compute.Close()
		platform.compute = nil
	}

	if platform.devicePool != nil {
		_ = platform.devicePool.Close()
		platform.devicePool = nil
	}

	return nil
}

func resolveComputeLocation(deviceName string) (tensor.Location, error) {
	switch deviceName {
	case "", "host", "cpu":
		return tensor.Host, nil
	case "metal":
		return tensor.Metal, nil
	case "cuda":
		return tensor.CUDA, nil
	case "xla":
		return tensor.XLA, nil
	default:
		return "", fmt.Errorf("runtime platform: unsupported compute device %q", deviceName)
	}
}

func defaultPlannerBindings() ir.SymbolMap {
	return ir.SymbolMap{
		"B": 1,
		"T": 1,
	}
}

func plannerBindingsForProgram(programPath string) ir.SymbolMap {
	bindings := defaultPlannerBindings()

	programYAML, err := readProgramBytes(programPath)

	if err != nil {
		return bindings
	}

	programParser := parse.NewParser()
	programAST, err := programParser.Program(programYAML)

	if err != nil {
		return bindings
	}

	for _, state := range programAST.State {
		if state.Name == "write_page_ids" && len(state.Shape) >= 1 {
			if tokenCount, ok := int64FromAny(state.Shape[0]); ok {
				bindings["N"] = tokenCount
				bindings["T"] = tokenCount
			}
		}

		if state.Type == "tensor" && len(state.Shape) >= 2 {
			if batchSize, ok := int64FromAny(state.Shape[0]); ok && batchSize > 0 {
				bindings["B"] = batchSize
			}

			if sequenceLength, ok := int64FromAny(state.Shape[1]); ok && sequenceLength > 0 {
				if state.Name == "latents" {
					bindings["T"] = sequenceLength
				}

				if state.Name == "text_embedding" {
					bindings["N"] = sequenceLength
					bindings["C"] = sequenceLength
				}
			}
		}

		if state.Type != "paged_tensor" {
			continue
		}

		if len(state.Shape) >= 5 {
			if layerCount, ok := int64FromAny(state.Shape[0]); ok {
				bindings["L"] = layerCount
			}

			if pageCount, ok := int64FromAny(state.Shape[1]); ok {
				bindings["P"] = pageCount
			}

			if pageSize, ok := int64FromAny(state.Shape[2]); ok {
				bindings["S"] = pageSize
			}

			if kvHeads, ok := int64FromAny(state.Shape[3]); ok {
				bindings["KVH"] = kvHeads
			}

			if headDim, ok := int64FromAny(state.Shape[4]); ok {
				bindings["HD"] = headDim
			}
		}

		if state.Config != nil {
			if pageCount, ok := int64FromAny(state.Config["page_count"]); ok {
				bindings["P"] = pageCount
			}

			if pageSize, ok := int64FromAny(state.Config["page_size"]); ok {
				bindings["S"] = pageSize
			}
		}
	}

	if pageCount, havePages := bindings["P"]; havePages {
		if pageSize, haveSize := bindings["S"]; haveSize && pageSize > 0 {
			bindings["KV"] = pageCount * pageSize
		}
	}

	return bindings
}

func int64FromAny(value any) (int64, bool) {
	switch typed := value.(type) {
	case int:
		return int64(typed), true
	case int32:
		return int64(typed), true
	case int64:
		return typed, true
	case float64:
		return int64(typed), true
	default:
		return 0, false
	}
}

func loadProgramWeights(
	ctx context.Context,
	programPath string,
	includeResolver compiler.IncludeResolver,
	resolveHub resolve.Hub,
	hubConfig *hub.HubConfig,
	memory tensor.Backend,
) (types.Parser, *execution.ResidentStore, error) {
	programYAML, err := readProgramBytes(programPath)

	if err != nil {
		return nil, nil, err
	}

	programParser := parse.NewParser()
	programAST, err := programParser.Program(programYAML)

	if err != nil {
		return nil, nil, fmt.Errorf("parse program: %w", err)
	}

	if len(programAST.Includes) == 0 {
		return nil, execution.NewResidentStore(memory), nil
	}

	weightStore := execution.NewResidentStore(memory)
	weightParsers := make([]types.Parser, 0, len(programAST.Includes))

	for includeName, includeSource := range programAST.Includes {
		repoID, component, ok := compiler.ParseHFReference(includeSource)

		if !ok {
			continue
		}

		blockYAML, err := includeResolver.ResolveInclude(ctx, compiler.IncludeSource{
			Name:   includeName,
			Source: includeSource,
		})

		if err != nil {
			return nil, nil, fmt.Errorf("resolve include %q: %w", includeName, err)
		}

		block, err := parse.BlockModelFromYAML(blockYAML)

		if err != nil {
			return nil, nil, fmt.Errorf("parse include block %q: %w", includeName, err)
		}

		subfolder := block.WeightSubfolder()

		if subfolder == "" && component != "" {
			subfolder = component
		}

		location := hub.ManifestRepoLocation(repoID, "", hubConfig.Token)
		bundle, _, err := execution.DownloadSafetensorsBundle(
			ctx,
			resolveHub,
			location,
			subfolder,
			hubConfig.CacheDir,
			memory,
		)

		if err != nil {
			return nil, nil, fmt.Errorf("download weights for include %q: %w", includeName, err)
		}

		weightStore.Absorb(bundle.Store)
		weightParsers = append(weightParsers, bundle.Parser)
	}

	return execution.NewMergedParser(weightParsers...), weightStore, nil
}

func readProgramBytes(programPath string) ([]byte, error) {
	if programPath == "" {
		return nil, fmt.Errorf("program path is required")
	}

	if filepath.IsAbs(programPath) {
		return os.ReadFile(programPath)
	}

	if _, err := os.Stat(programPath); err == nil {
		return os.ReadFile(programPath)
	}

	return asset.ReadFile(programPath)
}

func newSafetensorsParser(archive []byte) (types.Parser, error) {
	return safetensors.NewParser(archive)
}
