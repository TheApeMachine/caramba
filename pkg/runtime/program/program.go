package program

import "fmt"

/*
OperationID identifies a runtime operation registered with the
op registry. The family prefix ("io", "control", "graph", ...) is
preserved so the executor can route to the right family handler
without parsing strings repeatedly.
*/
type OperationID string

/*
Step is the typed runtime IR node. Inputs and Outputs are value
references; Config carries non-tensor scalar parameters; Body holds
nested steps for control-flow operations.
*/
type Step struct {
	ID      string
	Op      OperationID
	Inputs  map[string]ValueRef
	Outputs map[string]ValueRef
	Config  map[string]any
	Body    []Step
}

/*
Validate checks structural invariants on a single step. It does not
resolve references — that requires the surrounding program context
and runs in program.Validate.
*/
func (step Step) Validate() error {
	if step.ID == "" {
		return fmt.Errorf("runtime/program: step missing id")
	}

	if step.Op == "" {
		return fmt.Errorf("runtime/program: step %q missing op", step.ID)
	}

	for _, child := range step.Body {
		if err := child.Validate(); err != nil {
			return fmt.Errorf("step %q: %w", step.ID, err)
		}
	}

	return nil
}

/*
Walk visits this step then every descendant in pre-order. Returning
an error from visit halts the walk and propagates the error up.
*/
func (step Step) Walk(visit func(Step) error) error {
	if err := visit(step); err != nil {
		return err
	}

	for _, child := range step.Body {
		if err := child.Walk(visit); err != nil {
			return err
		}
	}

	return nil
}

/*
Program is the compiled runtime IR. It is the single artifact the
executor consumes; all manifest parsing, includes, repeats, and
typing happen in pkg/runtime/compiler before this point.
*/
type Program struct {
	Name        string
	Entry       string
	Backend     string
	Assets      []AssetDeclaration
	State       []StateDeclaration
	Samplers    []SamplerDeclaration
	Schedulers  []SchedulerDeclaration
	Graphs      map[string]GraphModule
	Steps       []Step
	Required    []CapabilityRequirement
	Provenance  ProvenanceDeclaration
	SourcePaths []string
}

/*
FindStep returns the first step in the program whose ID matches the
argument, searching nested bodies recursively. It returns nil when
no match exists.
*/
func (program *Program) FindStep(stepID string) *Step {
	for index := range program.Steps {
		if found := findStepIn(&program.Steps[index], stepID); found != nil {
			return found
		}
	}

	return nil
}

/*
StateByID returns the declaration for a named state object or nil.
*/
func (program *Program) StateByID(stateID string) *StateDeclaration {
	for index := range program.State {
		if program.State[index].ID == stateID {
			return &program.State[index]
		}
	}

	return nil
}

/*
AssetByID returns the declaration for a named asset or nil.
*/
func (program *Program) AssetByID(assetID string) *AssetDeclaration {
	for index := range program.Assets {
		if program.Assets[index].ID == assetID {
			return &program.Assets[index]
		}
	}

	return nil
}

/*
SamplerByID returns the declaration for a named sampler or nil.
*/
func (program *Program) SamplerByID(samplerID string) *SamplerDeclaration {
	for index := range program.Samplers {
		if program.Samplers[index].ID == samplerID {
			return &program.Samplers[index]
		}
	}

	return nil
}

/*
SchedulerByID returns the declaration for a named scheduler or nil.
*/
func (program *Program) SchedulerByID(schedulerID string) *SchedulerDeclaration {
	for index := range program.Schedulers {
		if program.Schedulers[index].ID == schedulerID {
			return &program.Schedulers[index]
		}
	}

	return nil
}

func findStepIn(step *Step, stepID string) *Step {
	if step.ID == stepID {
		return step
	}

	for index := range step.Body {
		if found := findStepIn(&step.Body[index], stepID); found != nil {
			return found
		}
	}

	return nil
}
