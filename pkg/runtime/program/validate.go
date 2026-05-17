package program

import "fmt"

/*
Validate runs strict reference checks across the entire program.
Every state, asset, sampler, scheduler, and graph reference inside
any step must resolve to a declared object. Step IDs must be unique
within their containing body. The executor relies on these invariants
holding before execution.
*/
func (program *Program) Validate() error {
	if program.Name == "" {
		return fmt.Errorf("runtime/program: program missing name")
	}

	for _, asset := range program.Assets {
		if err := asset.Validate(); err != nil {
			return err
		}
	}

	for _, state := range program.State {
		if err := state.Validate(); err != nil {
			return err
		}
	}

	for _, sampler := range program.Samplers {
		if err := sampler.Validate(); err != nil {
			return err
		}
	}

	for _, scheduler := range program.Schedulers {
		if err := scheduler.Validate(); err != nil {
			return err
		}
	}

	for graphID, graphModule := range program.Graphs {
		if graphModule.ID == "" {
			graphModule.ID = graphID
		}

		if err := graphModule.Validate(); err != nil {
			return err
		}
	}

	return program.validateSteps()
}

func (program *Program) validateSteps() error {
	if len(program.Steps) == 0 {
		return fmt.Errorf("runtime/program: program %q has no steps", program.Name)
	}

	for index := range program.Steps {
		if err := program.validateStepTree(&program.Steps[index], map[string]bool{}); err != nil {
			return err
		}
	}

	return nil
}

func (program *Program) validateStepTree(step *Step, seenIDs map[string]bool) error {
	if err := step.Validate(); err != nil {
		return err
	}

	if seenIDs[step.ID] {
		return fmt.Errorf(
			"runtime/program: duplicate step id %q in program %q",
			step.ID,
			program.Name,
		)
	}

	seenIDs[step.ID] = true

	if err := program.validateReferences(step); err != nil {
		return err
	}

	bodyIDs := map[string]bool{}

	for index := range step.Body {
		if err := program.validateStepTree(&step.Body[index], bodyIDs); err != nil {
			return err
		}
	}

	return nil
}

func (program *Program) validateReferences(step *Step) error {
	for _, ref := range step.Inputs {
		if err := program.checkRef(step.ID, ref); err != nil {
			return err
		}
	}

	for _, ref := range step.Outputs {
		if err := program.checkRef(step.ID, ref); err != nil {
			return err
		}
	}

	return nil
}

func (program *Program) checkRef(stepID string, ref ValueRef) error {
	switch ref.Namespace {
	case NamespaceLocal, NamespaceLiteral:
		return nil
	case NamespaceState:
		if program.StateByID(ref.Name) == nil {
			return fmt.Errorf(
				"runtime/program: step %q references undeclared state %q",
				stepID,
				ref.Name,
			)
		}

		return nil
	case NamespaceAsset:
		if program.AssetByID(ref.Name) == nil {
			return fmt.Errorf(
				"runtime/program: step %q references undeclared asset %q",
				stepID,
				ref.Name,
			)
		}

		return nil
	case NamespaceSampler:
		if program.SamplerByID(ref.Name) == nil {
			return fmt.Errorf(
				"runtime/program: step %q references undeclared sampler %q",
				stepID,
				ref.Name,
			)
		}

		return nil
	case NamespaceScheduler:
		if program.SchedulerByID(ref.Name) == nil {
			return fmt.Errorf(
				"runtime/program: step %q references undeclared scheduler %q",
				stepID,
				ref.Name,
			)
		}

		return nil
	case NamespaceGraph:
		if _, ok := program.Graphs[ref.Name]; !ok {
			return fmt.Errorf(
				"runtime/program: step %q references undeclared graph %q",
				stepID,
				ref.Name,
			)
		}

		return nil
	case NamespaceTokenizer, NamespaceDataset:
		return nil
	}

	return fmt.Errorf(
		"runtime/program: step %q has reference with unknown namespace %q",
		stepID,
		ref.Namespace,
	)
}
