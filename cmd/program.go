package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/hf/hub"
	"github.com/theapemachine/hf/program"
	"github.com/theapemachine/hf/safetensors"
	"github.com/theapemachine/manifesto/runtime"
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

	hubAdapter := hub.NewResolveAdapter(hub.NewClient(hubConfig))

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
	})

	if err != nil {
		return err
	}

	return programOrchestrator.Run(command.Context(), programPath)
}
