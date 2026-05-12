package manifest

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

	experiment.Datasets = extractDatasets(document)
	experiment.Trainer = extractOptionalMap(document, "trainer")
	experiment.Tuner = extractOptionalMap(document, "tuner")
	experiment.Deployment = extractOptionalMap(document, "deployment")

	rawTargets, ok := document["targets"].([]any)

	if ok {
		for _, rawTarget := range rawTargets {
			targetMap, ok := rawTarget.(map[string]any)

			if !ok {
				continue
			}

			target, err := compiler.buildTarget(targetMap)

			if err != nil {
				return nil, err
			}

			experiment.Targets = append(experiment.Targets, target)
		}
	}

	return experiment, nil
}

func (compiler *Compiler) buildTarget(targetMap map[string]any) (Target, error) {
	target := Target{
		Name:        stringField(targetMap, "name"),
		Description: stringField(targetMap, "description"),
		Backend:     stringField(targetMap, "backend"),
		Data:        extractOptionalMap(targetMap, "data"),
		System:      extractOptionalMap(targetMap, "system"),
	}

	systemBlock := target.System

	if topologyRaw, ok := systemBlock["topology"]; ok {
		topologyMap, ok := topologyRaw.(map[string]any)

		if ok {
			graph, err := compiler.buildGraph(topologyMap)

			if err != nil {
				return Target{}, err
			}

			target.Graph = graph
		}
	}

	rawRuns, ok := targetMap["runs"].([]any)

	if ok {
		for _, rawRun := range rawRuns {
			runMap, ok := rawRun.(map[string]any)

			if !ok {
				continue
			}

			target.Runs = append(target.Runs, buildRun(runMap))
		}
	}

	return target, nil
}

func buildRun(runMap map[string]any) Run {
	run := Run{
		ID:    stringField(runMap, "id"),
		Mode:  stringField(runMap, "mode"),
		Train: extractOptionalMap(runMap, "train"),
	}

	if seed, ok := runMap["seed"]; ok {
		run.Seed = anyInt(seed)
	}

	if steps, ok := runMap["steps"]; ok {
		run.Steps = anyInt(steps)
	}

	return run
}

func extractDatasets(document map[string]any) []map[string]any {
	raw, ok := document["datasets"].([]any)

	if !ok {
		return nil
	}

	out := make([]map[string]any, 0, len(raw))

	for _, item := range raw {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}

	return out
}

func extractOptionalMap(m map[string]any, key string) map[string]any {
	raw, ok := m[key]

	if !ok {
		return nil
	}

	result, ok := raw.(map[string]any)
	if !ok {
		return nil
	}

	return result
}

func stringField(m map[string]any, key string) string {
	v, ok := m[key].(string)
	if !ok {
		return ""
	}

	return v
}

func anyInt(v any) int {
	switch cast := v.(type) {
	case int:
		return cast
	case float64:
		return int(cast)
	default:
		return 0
	}
}
