package manifest

import "fmt"

/*
Run is a single named execution within a Target — one seed, one mode,
one set of training or evaluation parameters.
*/
type Run struct {
	ID    string
	Mode  string
	Seed  int
	Steps int
	Train map[string]any
}

/*
Target is a named experiment within a manifest. It declares a model system
(embedder + topology), a data reference, a backend, and one or more Runs.
*/
type Target struct {
	Name        string
	Description string
	Backend     string
	Data        map[string]any
	System      map[string]any
	Graph       *Graph
	Runs        []Run
}

/*
Experiment is the fully compiled output of a manifest. It carries all Targets
plus the shared top-level blocks (datasets, trainer, deployment) so that the
caller has a single object representing the complete research intent.
*/
type Experiment struct {
	Datasets   []map[string]any
	Trainer    map[string]any
	Tuner      map[string]any
	Deployment map[string]any
	Targets    []Target
}

/*
CompileExperiment parses the manifest at path and compiles all targets into
an Experiment. Each target's system.topology is built into a Graph.
*/
func (compiler *Compiler) CompileExperiment(path string) (*Experiment, error) {
	document, err := compiler.parser.Parse(path)

	if err != nil {
		return nil, err
	}

	experiment := &Experiment{}

	experiment.Datasets, err = extractDatasets(document)

	if err != nil {
		return nil, err
	}

	experiment.Trainer, err = extractOptionalMap(document, "trainer")

	if err != nil {
		return nil, err
	}

	experiment.Tuner, err = extractOptionalMap(document, "tuner")

	if err != nil {
		return nil, err
	}

	experiment.Deployment, err = extractOptionalMap(document, "deployment")

	if err != nil {
		return nil, err
	}

	rawTargets, err := compiler.requireSequence(document, "targets")

	if err != nil {
		return nil, err
	}

	for targetIndex, rawTarget := range rawTargets {
		targetMap, ok := rawTarget.(map[string]any)

		if !ok {
			return nil, fmt.Errorf("experiment: targets[%d] must be a mapping, got %T", targetIndex, rawTarget)
		}

		target, err := compiler.buildTarget(targetIndex, targetMap)

		if err != nil {
			return nil, err
		}

		experiment.Targets = append(experiment.Targets, target)
	}

	return experiment, nil
}

func (compiler *Compiler) buildTarget(targetIndex int, targetMap map[string]any) (Target, error) {
	path := fmt.Sprintf("targets[%d]", targetIndex)
	name, err := optionalStringField(targetMap, "name", path)

	if err != nil {
		return Target{}, err
	}

	description, err := optionalStringField(targetMap, "description", path)

	if err != nil {
		return Target{}, err
	}

	backend, err := optionalStringField(targetMap, "backend", path)

	if err != nil {
		return Target{}, err
	}

	data, err := extractOptionalMapAt(targetMap, "data", path)

	if err != nil {
		return Target{}, err
	}

	system, err := extractOptionalMapAt(targetMap, "system", path)

	if err != nil {
		return Target{}, err
	}

	target := Target{
		Name:        name,
		Description: description,
		Backend:     backend,
		Data:        data,
		System:      system,
	}

	systemBlock := target.System

	if systemBlock == nil {
		return Target{}, fmt.Errorf("experiment: %s.system is required", path)
	}

	topologyRaw, ok := systemBlock["topology"]

	if !ok {
		return Target{}, fmt.Errorf("experiment: %s.system.topology is required", path)
	}

	topologyMap, ok := topologyRaw.(map[string]any)

	if !ok {
		return Target{}, fmt.Errorf("experiment: %s.system.topology must be a mapping, got %T", path, topologyRaw)
	}

	graph, err := compiler.buildGraph(topologyMap)

	if err != nil {
		return Target{}, fmt.Errorf("experiment: %s.system.topology: %w", path, err)
	}

	target.Graph = graph
	rawRuns, err := requireOptionalSequenceAt(targetMap, "runs", path)

	if err != nil {
		return Target{}, err
	}

	for runIndex, rawRun := range rawRuns {
		runMap, ok := rawRun.(map[string]any)

		if !ok {
			return Target{}, fmt.Errorf("experiment: %s.runs[%d] must be a mapping, got %T", path, runIndex, rawRun)
		}

		run, err := buildRun(fmt.Sprintf("%s.runs[%d]", path, runIndex), runMap)

		if err != nil {
			return Target{}, err
		}

		target.Runs = append(target.Runs, run)
	}

	return target, nil
}

func buildRun(path string, runMap map[string]any) (Run, error) {
	id, err := optionalStringField(runMap, "id", path)

	if err != nil {
		return Run{}, err
	}

	mode, err := optionalStringField(runMap, "mode", path)

	if err != nil {
		return Run{}, err
	}

	train, err := extractOptionalMapAt(runMap, "train", path)

	if err != nil {
		return Run{}, err
	}

	run := Run{
		ID:    id,
		Mode:  mode,
		Train: train,
	}

	if seed, ok := runMap["seed"]; ok {
		parsedSeed, err := strictInt(seed)

		if err != nil {
			return Run{}, fmt.Errorf("experiment: %s.seed: %w", path, err)
		}

		run.Seed = parsedSeed
	}

	if steps, ok := runMap["steps"]; ok {
		parsedSteps, err := strictInt(steps)

		if err != nil {
			return Run{}, fmt.Errorf("experiment: %s.steps: %w", path, err)
		}

		run.Steps = parsedSteps
	}

	return run, nil
}

func extractDatasets(document map[string]any) ([]map[string]any, error) {
	rawValue, ok := document["datasets"]

	if !ok {
		return nil, nil
	}

	raw, ok := rawValue.([]any)

	if !ok {
		return nil, fmt.Errorf("experiment: datasets must be a sequence, got %T", rawValue)
	}

	out := make([]map[string]any, 0, len(raw))

	for index, item := range raw {
		m, ok := item.(map[string]any)

		if !ok {
			return nil, fmt.Errorf("experiment: datasets[%d] must be a mapping, got %T", index, item)
		}

		out = append(out, m)
	}

	return out, nil
}

func extractOptionalMap(m map[string]any, key string) (map[string]any, error) {
	return extractOptionalMapAt(m, key, "experiment")
}

func extractOptionalMapAt(m map[string]any, key string, path string) (map[string]any, error) {
	raw, ok := m[key]

	if !ok {
		return nil, nil
	}

	result, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("experiment: %s.%s must be a mapping, got %T", path, key, raw)
	}

	return result, nil
}

func requireOptionalSequenceAt(m map[string]any, key string, path string) ([]any, error) {
	raw, ok := m[key]

	if !ok {
		return nil, nil
	}

	sequence, ok := raw.([]any)

	if !ok {
		return nil, fmt.Errorf("experiment: %s.%s must be a sequence, got %T", path, key, raw)
	}

	return sequence, nil
}

func optionalStringField(m map[string]any, key string, path string) (string, error) {
	v, ok := m[key]

	if !ok {
		return "", nil
	}

	text, ok := v.(string)

	if !ok {
		return "", fmt.Errorf("experiment: %s.%s must be a string, got %T", path, key, v)
	}

	return text, nil
}

func strictInt(v any) (int, error) {
	switch cast := v.(type) {
	case int:
		return cast, nil
	case float64:
		if cast != float64(int(cast)) {
			return 0, fmt.Errorf("must be an integer, got %v", cast)
		}

		return int(cast), nil
	default:
		return 0, fmt.Errorf("must be an integer, got %T", v)
	}
}
