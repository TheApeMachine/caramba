package diffusion

import (
	"context"
	"fmt"
	"os"

	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/hf/hub"
	"github.com/theapemachine/hf/program"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/compiler"
	"github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/pool"
	"github.com/theapemachine/puter/runner"
	"github.com/theapemachine/qpool"
)

const (
	denoiserGraphName    = "flux2klein"
	textEncoderGraphName = "flux2klein.text_encoder"
	schedulerName        = "scheduler"
)

/*
Harness owns a compiled diffusion program session for diagnostic graph calls.
*/
type Harness struct {
	Session     *runtime.ProgramSession
	Scheduler   *runtime.FlowMatchEulerDiscrete
	StateMemory tensor.Backend
	Compile     *compiler.CompileOutput
	GraphRunner *runner.Runner
	devicePool  *pool.Pool
	workerPool  *qpool.Q
	zeroPrompt  tensor.Tensor
}

/*
NewHarness compiles runtime/diffusion-diagnose.yml (no VAE), materializes state, and encodes the prompt.
*/
func NewHarness(ctx context.Context, prompt string) (*Harness, error) {
	hubConfig := config.NewHubConfig()
	qpoolConfig := config.NewQPoolConfig()

	workerPool := qpoolConfig.NewWorkerPool(ctx)

	devicePool, err := pool.New(ctx, workerPool)

	if err != nil {
		return nil, fmt.Errorf("diffusion harness: discover devices: %w", err)
	}

	stateMemory, _, err := devicePool.MemoryBackend()

	if err != nil {
		devicePool.Close()
		return nil, fmt.Errorf("diffusion harness: state memory: %w", err)
	}

	graphRunner := runner.New(devicePool)

	output, err := runtime.CompileProgramFromAsset(
		ctx,
		diagnoseProgramPath,
		hub.NewResolveAdapter(hub.NewClient(hubConfig)),
		hubConfig.CacheDir,
	)

	if err != nil {
		graphRunner.Close()
		devicePool.Close()
		return nil, err
	}

	session, err := runtime.NewProgramSession(runtime.ProgramSessionOptions{
		Program:      output.Program,
		Graphs:       output.Graphs,
		Compute:      output.ComputeGraphs,
		Backend:      graphRunner,
		Host:         program.NewHost(program.HostOptions{Stdin: os.Stdin, HubConfig: hubConfig}),
		StateBackend: stateMemory,
	})

	if err != nil {
		graphRunner.Close()
		devicePool.Close()
		return nil, err
	}

	scheduler, err := session.FlowMatchScheduler(schedulerName)

	if err != nil {
		graphRunner.Close()
		devicePool.Close()
		return nil, err
	}

	prefix, err := encodePromptPrefix(output.Program)

	if err != nil {
		graphRunner.Close()
		devicePool.Close()
		return nil, err
	}

	if err := session.RunSteps(ctx, prefix, map[string]any{"prompt": prompt}); err != nil {
		graphRunner.Close()
		devicePool.Close()
		return nil, fmt.Errorf("diffusion harness: encode prompt: %w", err)
	}

	return &Harness{
		Session:     session,
		Scheduler:   scheduler,
		StateMemory: stateMemory,
		Compile:     output,
		GraphRunner: graphRunner,
		devicePool:  devicePool,
		workerPool:  workerPool,
	}, nil
}

/*
Close releases runner and device pool resources owned by the harness.
*/
func (harness *Harness) Close() {
	if harness == nil {
		return
	}

	if harness.GraphRunner != nil {
		harness.GraphRunner.Close()
	}

	if harness.devicePool != nil {
		harness.devicePool.Close()
	}

	if harness.zeroPrompt != nil {
		_ = harness.zeroPrompt.Close()
	}

	if harness.workerPool != nil {
		harness.workerPool.Close()
	}
}

func encodePromptPrefix(program *ast.Program) ([]ast.Step, error) {
	if program == nil {
		return nil, fmt.Errorf("diffusion harness: program is required")
	}

	if len(program.Steps) < 2 {
		return nil, fmt.Errorf("diffusion harness: program needs tokenizer and text encoder steps")
	}

	return program.Steps[:2], nil
}
