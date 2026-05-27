package cmd

import (
	"fmt"
	"io"
	"sort"
	"text/tabwriter"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/hf/hub"
	hfprogram "github.com/theapemachine/hf/program"
	"github.com/theapemachine/manifesto/asset"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/catalog"
	"github.com/theapemachine/manifesto/compiler"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/typer"
)

var diffusionDiagnoseCmd = &cobra.Command{
	Use:   "diffusion-diagnose",
	Short: "Compile the diffusion runtime and print workspace diagnostics.",
	RunE: func(command *cobra.Command, args []string) error {
		return runProgramDiagnostic(command, "runtime/diffusion.yml")
	},
}

func init() {
	rootCmd.AddCommand(diffusionDiagnoseCmd)
}

func runProgramDiagnostic(command *cobra.Command, programPath string) error {
	hubConfig := config.NewHubConfig()
	hubClient := hub.NewClient(hubConfig)
	hubAdapter := hub.NewResolveAdapter(hubClient)

	includeResolver, err := hfprogram.NewIncludeResolver(hfprogram.IncludeResolverOptions{
		Hub:      hubAdapter,
		CacheDir: hubConfig.CacheDir,
	})

	if err != nil {
		return fmt.Errorf("caramba: build include resolver: %w", err)
	}

	programYAML, err := asset.ReadFile(programPath)

	if err != nil {
		return fmt.Errorf("caramba: read program %q: %w", programPath, err)
	}

	manifestCompiler, err := compiler.NewProgramCompiler(
		compiler.NewPool(catalog.NewFS(asset.TemplateFS())),
	)

	if err != nil {
		return fmt.Errorf("caramba: new compiler: %w", err)
	}

	manifestCompiler = manifestCompiler.
		WithIncludeResolver(includeResolver).
		WithTyperOptions(typer.Options{DisableSynthesis: true}).
		WithPlannerBindings(defaultPlannerBindings())

	output, err := manifestCompiler.CompileAssets(command.Context(), compiler.CompileInput{
		ProgramYAML: programYAML,
		CacheDir:    hubConfig.CacheDir,
	}, asset.TemplateFS())

	if err != nil {
		return fmt.Errorf("caramba: compile program %q: %w", programPath, err)
	}

	return writeWorkspaceReport(command.OutOrStdout(), programPath, output)
}

func writeWorkspaceReport(
	writer io.Writer,
	programPath string,
	output *compiler.CompileOutput,
) error {
	if output == nil {
		return fmt.Errorf("caramba: compile output is required")
	}

	graphNames := sortedWorkspaceNames(output.Workspaces)
	table := tabwriter.NewWriter(writer, 0, 0, 2, ' ', 0)

	fmt.Fprintf(table, "program\t%s\n", programPath)
	fmt.Fprintf(table, "graphs\t%d\n", len(output.Graphs))
	fmt.Fprintln(table)
	fmt.Fprintln(table, "graph\tnodes\tworkspace\tallocations\tlargest interval")

	for _, name := range graphNames {
		topology := output.Workspaces[name]
		graph := output.Graphs[name]
		fmt.Fprintf(
			table,
			"%s\t%d\t%s\t%d\t%s\n",
			name,
			graphNodeCount(graph),
			formatByteCount(topology.Workspace.Size),
			len(topology.Workspace.Allocations),
			formatByteCount(largestIntervalSize(topology)),
		)
	}

	return table.Flush()
}

func sortedWorkspaceNames(workspaces map[string]*ir.Topology) []string {
	names := make([]string, 0, len(workspaces))

	for name := range workspaces {
		names = append(names, name)
	}

	sort.Strings(names)

	return names
}

func graphNodeCount(graph *ast.Graph) int {
	if graph == nil {
		return 0
	}

	return len(graph.Nodes)
}

func largestIntervalSize(topology *ir.Topology) int64 {
	if topology == nil {
		return 0
	}

	var largest int64

	for _, interval := range topology.Workspace.Allocations {
		if interval.Size <= largest {
			continue
		}

		largest = interval.Size
	}

	return largest
}

func formatByteCount(bytes int64) string {
	const unit = int64(1024)

	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}

	value := float64(bytes)
	units := []string{"KiB", "MiB", "GiB", "TiB"}

	for _, suffix := range units {
		value /= float64(unit)

		if value < float64(unit) {
			return fmt.Sprintf("%.2f %s", value, suffix)
		}
	}

	return fmt.Sprintf("%.2f PiB", value/float64(unit))
}
