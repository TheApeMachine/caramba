package compiler

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/manifest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
Compiler turns a manifest document into a runtime program.Program.
It reuses the existing manifest parser for YAML loading, include
resolution, variable interpolation, and repeat expansion; the
compiler's job is to interpret the system.runtime block, normalize
the YAML shortcuts described in the platform requirements, and
return a typed, fully validated runtime IR.
*/
type Compiler struct {
	parser *manifest.Parser
}

/*
New constructs a Compiler anchored at projectRoot. The same Parser
used by the static manifest compiler is used here so includes and
variables resolve identically.
*/
func New(projectRoot string) *Compiler {
	return &Compiler{parser: manifest.NewParser(projectRoot)}
}

/*
Compile loads the manifest at relativePath and returns the
runtime.Program declared under system.runtime. Validation runs at
the end so a returned program is always safe to hand to the
executor.
*/
func (compiler *Compiler) Compile(relativePath string) (*program.Program, error) {
	document, err := compiler.parser.Parse(relativePath)

	if err != nil {
		return nil, fmt.Errorf("compiler: parsing %s: %w", relativePath, err)
	}

	runtimeProgram, err := compiler.CompileDocument(document)

	if err != nil {
		return nil, err
	}

	runtimeProgram.SourcePaths = append(runtimeProgram.SourcePaths, relativePath)

	return runtimeProgram, nil
}

/*
CompileBytes parses raw YAML and returns the compiled program.
*/
func (compiler *Compiler) CompileBytes(data []byte) (*program.Program, error) {
	document, err := compiler.parser.ParseBytes(data)

	if err != nil {
		return nil, fmt.Errorf("compiler: parsing bytes: %w", err)
	}

	return compiler.CompileDocument(document)
}

/*
CompileDocument compiles an already-parsed document. It is the
single entry point both Compile and CompileBytes go through.
*/
func (compiler *Compiler) CompileDocument(document map[string]any) (*program.Program, error) {
	runtimeBlock, err := yamlMap(yamlPath(document, "system", "runtime"), "system.runtime")

	if err != nil {
		return nil, err
	}

	if runtimeBlock == nil {
		return nil, fmt.Errorf("compiler: document has no system.runtime block")
	}

	if err := requireProgramType(runtimeBlock); err != nil {
		return nil, err
	}

	runtimeProgram := &program.Program{
		Graphs: map[string]program.GraphModule{},
	}

	if name, ok := runtimeBlock["name"]; ok {
		runtimeProgram.Name, err = yamlString(name, "system.runtime.name")

		if err != nil {
			return nil, err
		}
	}

	if runtimeProgram.Name == "" {
		runtimeProgram.Name, _ = yamlString(yamlPath(document, "name"), "name")
	}

	if runtimeProgram.Name == "" {
		runtimeProgram.Name = "program"
	}

	if entry, ok := runtimeBlock["entry"]; ok {
		runtimeProgram.Entry, err = yamlString(entry, "system.runtime.entry")

		if err != nil {
			return nil, err
		}
	}

	if backend, ok := runtimeBlock["backend"]; ok {
		runtimeProgram.Backend, err = yamlString(backend, "system.runtime.backend")

		if err != nil {
			return nil, err
		}
	}

	if err := compileAssets(runtimeBlock, runtimeProgram); err != nil {
		return nil, err
	}

	if err := compileState(runtimeBlock, runtimeProgram); err != nil {
		return nil, err
	}

	if err := compileSamplers(runtimeBlock, runtimeProgram); err != nil {
		return nil, err
	}

	if err := compileSchedulers(runtimeBlock, runtimeProgram); err != nil {
		return nil, err
	}

	if err := compileGraphs(runtimeBlock, runtimeProgram); err != nil {
		return nil, err
	}

	if err := compileSteps(runtimeBlock, runtimeProgram); err != nil {
		return nil, err
	}

	if err := runtimeProgram.Validate(); err != nil {
		return nil, fmt.Errorf("compiler: validating program %q: %w", runtimeProgram.Name, err)
	}

	return runtimeProgram, nil
}

func requireProgramType(runtimeBlock map[string]any) error {
	raw, ok := runtimeBlock["type"]

	if !ok {
		return nil
	}

	typed, ok := raw.(string)

	if !ok {
		return fmt.Errorf("compiler: system.runtime.type must be a string, got %T", raw)
	}

	if typed != "program" {
		return fmt.Errorf(
			"compiler: system.runtime.type must be 'program', got %q",
			typed,
		)
	}

	return nil
}
