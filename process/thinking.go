package process

// import (
// 	"encoding/json"
// 	"time"

// 	"github.com/theapemachine/amsh/utils"
// )

// /*
// Thinking is a process that allows the system to think about a given topic.
// It now includes a detailed reasoning graph to capture multi-level and interconnected reasoning steps.
// */
// type Thinking struct {
// 	HypergraphLayer     HypergraphLayer     `json:"hypergraph_layer" jsonschema:"title=HypergraphLayer,description=Represents many-to-many relationships and group dynamics,required"`
// 	TensorNetwork       TensorNetwork       `json:"tensor_network" jsonschema:"title=TensorNetwork,description=Multi-dimensional relationship patterns,required"`
// 	FractalStructure    FractalStructure    `json:"fractal_structure" jsonschema:"title=FractalStructure,description=Self-similar patterns at different scales,required"`
// 	QuantumLayer        QuantumLayer        `json:"quantum_layer" jsonschema:"title=QuantumLayer,description=Probabilistic and superposition states,required"`
// 	HolographicMemory   HolographicMemory   `json:"holographic_memory" jsonschema:"title=HolographicMemory,description=Distributed information storage,required"`
// 	TemporalDynamics    TemporalDynamics    `json:"temporal_dynamics" jsonschema:"title=TemporalDynamics,description=Time-based evolution of thoughts,required"`
// 	EmergentPatterns    EmergentPatterns    `json:"emergent_patterns" jsonschema:"title=EmergentPatterns,description=Higher-order patterns that emerge from interactions,required"`
// 	CrossLayerSynthesis CrossLayerSynthesis `json:"cross_layer_synthesis" jsonschema:"title=CrossLayerSynthesis,description=Integration across different representation layers,required"`
// 	UnifiedPerspective  UnifiedPerspective  `json:"unified_perspective" jsonschema:"title=UnifiedPerspective,description=Coherent view across all structures,required"`
// }

// type Conflict struct {
// 	ID         string   `json:"id" jsonschema:"required,title=ID,description=Unique identifier for the conflict"`
// 	Elements   []string `json:"elements" jsonschema:"required,title=Elements,description=Elements in conflict"`
// 	Type       string   `json:"type" jsonschema:"required,title=Type,description=Type of conflict"`
// 	Severity   float64  `json:"severity" jsonschema:"required,title=Severity,description=Severity of the conflict"`
// 	Resolution string   `json:"resolution" jsonschema:"required,title=Resolution,description=Resolution of the conflict"`
// }

// type UnifiedInsight struct {
// 	ID           string   `json:"id" jsonschema:"required,title=ID,description=Unique identifier for the insight"`
// 	Description  string   `json:"description" jsonschema:"required,title=Description,description=Description of the insight"`
// 	Sources      []string `json:"sources" jsonschema:"required,title=Sources,description=Sources of the insight"`
// 	Confidence   float64  `json:"confidence" jsonschema:"required,title=Confidence,description=Confidence in the insight"`
// 	Impact       float64  `json:"impact" jsonschema:"required,title=Impact,description=Impact of the insight"`
// 	Applications []string `json:"applications" jsonschema:"required,title=Applications,description=Applications of the insight"`
// }

// // Helper types
// type Properties map[string]interface{}

// type Metrics struct {
// 	Coherence  float64 `json:"coherence" jsonschema:"required,title=Coherence,description=Coherence metric"`
// 	Complexity float64 `json:"complexity" jsonschema:"required,title=Complexity,description=Complexity metric"`
// 	Stability  float64 `json:"stability" jsonschema:"required,title=Stability,description=Stability metric"`
// 	Novelty    float64 `json:"novelty" jsonschema:"required,title=Novelty,description=Novelty metric"`
// }

// type ProcessResult struct {
// 	CoreID string          `json:"core_id" jsonschema:"required,title=Core ID,description=Core ID,"`
// 	Data   json.RawMessage `json:"data" jsonschema:"title=Data,description=Data from the core"`
// 	Error  error           `json:"error" jsonschema:"title=Error,description=Error from the core"`
// }

// // Integration type for final results
// type ThinkingResult struct {
// 	Surface SurfaceAnalysis `json:"surface" jsonschema:"required,title=Surface,description=Surface analysis"`
// 	Pattern PatternAnalysis `json:"pattern" jsonschema:"required,title=Pattern,description=Pattern analysis"`
// 	Quantum QuantumAnalysis `json:"quantum" jsonschema:"required,title=Quantum,description=Quantum analysis"`
// 	Time    TimeAnalysis    `json:"time" jsonschema:"required,title=Time,description=Time analysis"`
// }

// func (thinking *Thinking) SystemPrompt(key string) string {
// 	return utils.SystemPrompt(key, "thinking", utils.GenerateSchema[Thinking]())
// }
