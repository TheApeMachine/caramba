package program

import "fmt"

/*
StateDeclaration names a runtime state object and the type that
should be instantiated by the state registry. Backend pins the state
to a compute device when required (kv_cache, tensor, optimizer).
Config carries the typed parameters the state implementation needs
to initialize itself.
*/
type StateDeclaration struct {
	ID      string
	Type    string
	Backend string
	Config  map[string]any
}

/*
Validate enforces that a state declaration is internally consistent
before the executor builds it.
*/
func (stateDeclaration StateDeclaration) Validate() error {
	if stateDeclaration.ID == "" {
		return fmt.Errorf("runtime/program: state declaration missing id")
	}

	if stateDeclaration.Type == "" {
		return fmt.Errorf("runtime/program: state %q missing type", stateDeclaration.ID)
	}

	return nil
}

/*
AssetDeclaration names an external artifact the program depends on
(model weights, tokenizer, dataset). The runtime resolves Source via
pkg/asset; Config carries loader-specific options.
*/
type AssetDeclaration struct {
	ID     string
	Kind   string
	Source string
	Config map[string]any
}

func (assetDeclaration AssetDeclaration) Validate() error {
	if assetDeclaration.ID == "" {
		return fmt.Errorf("runtime/program: asset declaration missing id")
	}

	if assetDeclaration.Kind == "" {
		return fmt.Errorf("runtime/program: asset %q missing kind", assetDeclaration.ID)
	}

	if assetDeclaration.Source == "" {
		return fmt.Errorf("runtime/program: asset %q missing source", assetDeclaration.ID)
	}

	return nil
}

/*
SamplerDeclaration configures a runtime sampler that the program
will reuse across decode steps.
*/
type SamplerDeclaration struct {
	ID     string
	Type   string
	Config map[string]any
}

func (samplerDeclaration SamplerDeclaration) Validate() error {
	if samplerDeclaration.ID == "" {
		return fmt.Errorf("runtime/program: sampler declaration missing id")
	}

	if samplerDeclaration.Type == "" {
		return fmt.Errorf("runtime/program: sampler %q missing type", samplerDeclaration.ID)
	}

	return nil
}

/*
SchedulerDeclaration configures a diffusion-style scheduler whose
state is mutated by scheduler.step.
*/
type SchedulerDeclaration struct {
	ID     string
	Type   string
	Config map[string]any
}

func (schedulerDeclaration SchedulerDeclaration) Validate() error {
	if schedulerDeclaration.ID == "" {
		return fmt.Errorf("runtime/program: scheduler declaration missing id")
	}

	if schedulerDeclaration.Type == "" {
		return fmt.Errorf("runtime/program: scheduler %q missing type", schedulerDeclaration.ID)
	}

	return nil
}

/*
GraphModule references a compiled compute graph the runtime program
calls via graph.call. Manifest is the manifest source path; Topology
names the topology subtree inside that manifest. WeightAsset binds
the graph's parameters to a declared model asset.
*/
type GraphModule struct {
	ID          string
	Manifest    string
	Topology    string
	WeightAsset string
	Config      map[string]any
}

func (graphModule GraphModule) Validate() error {
	if graphModule.ID == "" {
		return fmt.Errorf("runtime/program: graph module missing id")
	}

	if graphModule.Manifest == "" && graphModule.Topology == "" {
		return fmt.Errorf(
			"runtime/program: graph %q needs either manifest or topology",
			graphModule.ID,
		)
	}

	return nil
}

/*
CapabilityRequirement is a contract entry the backend must satisfy
for the program to be legal. It mirrors the per-operation capability
contract the orchestrator already enforces for static graphs.
*/
type CapabilityRequirement struct {
	Operation string
	Dtype     string
	Layout    string
	Backend   string
	State     string
}

func (capabilityRequirement CapabilityRequirement) Validate() error {
	if capabilityRequirement.Operation == "" {
		return fmt.Errorf("runtime/program: capability requirement missing operation")
	}

	return nil
}

/*
ProvenanceDeclaration records what the runtime must capture into the
signed ledger for this program: manifest paths, asset hashes, seeds,
metrics, traces, and output artifacts.
*/
type ProvenanceDeclaration struct {
	Manifests []string
	Assets    []string
	Seeds     []string
	Metrics   []string
	Traces    []string
	Outputs   []string
}
