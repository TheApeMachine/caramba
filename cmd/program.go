package cmd

import (
	"context"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/hf/hub"
	"github.com/theapemachine/hf/program"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/puter/pool"
	"github.com/theapemachine/puter/runner"
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

	graphRunner := runner.New(devicePool)
	defer graphRunner.Close()

	programOrchestrator, err := runtime.NewOrchestrator(runtime.OrchestratorOptions{
		Hub:           hub.NewResolveAdapter(hub.NewClient(hubConfig)),
		Compute:       graphRunner,
		Host:          program.NewHost(program.HostOptions{Stdin: os.Stdin, HubConfig: hubConfig}),
		StateMemory:   stateMemory,
		CacheDir:      hubConfig.CacheDir,
		Stdin:         os.Stdin,
		InitialValues: initialValues,
	})

	if err != nil {
		return err
	}

	return programOrchestrator.Run(command.Context(), programPath)
}

func runProgramContext(
	ctx context.Context,
	programPath string,
	initialValues map[string]any,
) error {
	command := &cobra.Command{}
	command.SetContext(ctx)

	return runProgram(command, programPath, initialValues)
}
