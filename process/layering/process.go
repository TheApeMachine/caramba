package layering

type Workload struct {
	Name string `json:"name" jsonschema:"title=Name,description=The name of the workload,enum=temporal_dynamics,enum=holographic_memory,enum=fractal_structure,enum=hypergraph,enum=tensor_network,enum=quantum_layer,enum=ideation,enum=context_mapping,enum=story_flow,enum=research,enum=architecture,enum=requirements,enum=implementation,enum=testing,enum=deployment,enum=documentation,enum=review,required"`
}

type Layer struct {
	Workloads []Workload `json:"workloads" jsonschema:"title=Workloads,description=The workloads that should be processed for this layer,required"`
}

type Fork struct {
	Description string  `json:"description" jsonschema:"title=Description,description=A description of the fork,required"`
	Layers      []Layer `json:"layers" jsonschema:"title=Layers,description=The layers that should be involved in the processing of this fork,required"`
}

type Process struct {
	Layers []Layer `json:"layers" jsonschema:"title=Layers,description=The layers that should be involved in the processing of the incoming request,required"`
	Forks  []Fork  `json:"forks" jsonschema:"title=Forks,description=Alternative, complimentary, or competing paths to the main process"`
}

func NewProcess() *Process {
	return &Process{}
}
