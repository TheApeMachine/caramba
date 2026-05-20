package runtime

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/caramba/pkg/hub"
	manifest "github.com/theapemachine/manifesto"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/ir"
	manifestruntime "github.com/theapemachine/manifesto/runtime"
)

/*
Session compiles and executes one runtime program manifest.
*/
type Session struct {
	program      *ast.Program
	graphs       map[string]*ast.Graph
	compute      map[string]*ir.Graph
	graphBackend manifestruntime.Backend
	hostOps      manifestruntime.HostOps
	state        *manifestruntime.StateStore
	schedulers   map[string]*manifestruntime.FlowMatchEulerDiscrete
	stdin        io.Reader
}

/*
OpenSession compiles a runtime program and prepares execution resources.
*/
func OpenSession(
	ctx context.Context,
	programPath string,
) (*Session, error) {
	programYAML, err := asset.ReadFile(programPath)

	if err != nil {
		return nil, fmt.Errorf("runtime session: read program %q: %w", programPath, err)
	}

	hubConfig := config.NewHubConfig()
	hubClient := hub.NewClient(hubConfig)
	hubAdapter := hub.NewResolveAdapter(hubClient)

	compiler, err := manifest.NewCompiler(manifest.Options{
		Catalog: catalog.NewFS(asset.TemplateFS()),
		Hub:     hubAdapter,
	})

	if err != nil {
		return nil, err
	}

	output, err := compiler.CompileAssets(ctx, manifest.CompileInput{
		ProgramYAML: programYAML,
		CacheDir:    hubConfig.CacheDir,
	}, asset.TemplateFS())

	if err != nil {
		return nil, err
	}

	computeBackend, err := compute.NewBackend(ctx, nil)

	if err != nil {
		return nil, err
	}

	graphBackend, err := NewGraphBackend(computeBackend)

	if err != nil {
		return nil, err
	}

	stateStore, err := manifestruntime.NewStateStore(output.Program.State)

	if err != nil {
		return nil, err
	}

	schedulers, err := schedulersFromProgram(output.Program)

	if err != nil {
		return nil, err
	}

	return &Session{
		program:      output.Program,
		graphs:       output.Graphs,
		compute:      output.ComputeGraphs,
		graphBackend: graphBackend,
		hostOps:      NewCarambaHostOps(hubConfig),
		state:        stateStore,
		schedulers:   schedulers,
		stdin:        os.Stdin,
	}, nil
}

/*
Run executes the compiled program.
*/
func (session *Session) Run(ctx context.Context) error {
	return session.run(ctx, nil)
}

/*
RunWithValues executes the program with pre-populated value bindings.
*/
func (session *Session) RunWithValues(ctx context.Context, initial map[string]any) error {
	return session.run(ctx, initial)
}

func (session *Session) run(ctx context.Context, initial map[string]any) error {
	if session == nil {
		return fmt.Errorf("runtime session: session is required")
	}

	executor := manifestruntime.NewExecutor(manifestruntime.ExecutorOptions{
		Backend:       session.graphBackend,
		Host:          session.hostOps,
		State:         session.state,
		Schedulers:    session.schedulers,
		Stdin:         session.stdin,
		InitialValues: initial,
	})

	computeAny := make(map[string]any, len(session.compute))

	for name, graph := range session.compute {
		computeAny[name] = graph
	}

	return executor.Run(ctx, session.program, session.graphs, computeAny)
}

func schedulersFromProgram(program *ast.Program) (map[string]*manifestruntime.FlowMatchEulerDiscrete, error) {
	schedulers := make(map[string]*manifestruntime.FlowMatchEulerDiscrete)

	if program == nil {
		return schedulers, nil
	}

	for name, declaration := range program.Schedulers {
		switch declaration.Type {
		case "flow_match_euler_discrete":
			steps := intFromAny(declaration.Config["steps"], 28)
			trainSteps := intFromAny(declaration.Config["num_train_timesteps"], 1000)

			scheduler, err := manifestruntime.NewFlowMatchEulerDiscrete(manifestruntime.SchedulerConfig{
				Steps:             steps,
				NumTrainTimesteps: trainSteps,
			})

			if err != nil {
				return nil, err
			}

			schedulers[name] = scheduler
		default:
			return nil, fmt.Errorf("runtime session: unsupported scheduler type %q", declaration.Type)
		}
	}

	return schedulers, nil
}

func intFromAny(value any, fallback int) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return fallback
	}
}
