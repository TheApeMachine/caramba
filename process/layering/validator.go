package layering

import (
	"fmt"
)

type WorkloadType string

const (
	TypeSimulation WorkloadType = "simulation"
	TypeProcess    WorkloadType = "process"
)

// WorkloadRule defines characteristics and valid combinations
type WorkloadRule struct {
	Type        WorkloadType
	Category    string   // For grouping related workloads
	MinInLayer  int      // Minimum number of workloads of this type per layer
	MaxInLayer  int      // Maximum number of workloads of this type per layer
	ValidWith   []string // Other workloads this can be combined with
	Description string   // Helps provide meaningful feedback
}

// Validator handles process validation
type Validator struct {
	rules map[string]WorkloadRule
}

func NewValidator() *Validator {
	v := &Validator{
		rules: map[string]WorkloadRule{
			// Simulation workloads
			"temporal_dynamics": {
				Type:        TypeSimulation,
				Category:    "simulation",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"quantum_layer", "fractal_structure", "holographic_memory", "tensor_network", "hypergraph"},
				Description: "Temporal evolution modeling",
			},
			"quantum_layer": {
				Type:        TypeSimulation,
				Category:    "simulation",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"temporal_dynamics", "fractal_structure", "holographic_memory", "tensor_network", "hypergraph"},
				Description: "Multiple possibility handling",
			},
			"fractal_structure": {
				Type:        TypeSimulation,
				Category:    "simulation",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"temporal_dynamics", "quantum_layer", "holographic_memory", "tensor_network", "hypergraph"},
				Description: "Pattern consistency",
			},
			"holographic_memory": {
				Type:        TypeSimulation,
				Category:    "simulation",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"temporal_dynamics", "quantum_layer", "fractal_structure", "tensor_network", "hypergraph"},
				Description: "Distributed information",
			},
			"tensor_network": {
				Type:        TypeSimulation,
				Category:    "simulation",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"temporal_dynamics", "quantum_layer", "fractal_structure", "holographic_memory", "hypergraph"},
				Description: "Relationship modeling",
			},
			"hypergraph": {
				Type:        TypeSimulation,
				Category:    "simulation",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"temporal_dynamics", "quantum_layer", "fractal_structure", "holographic_memory", "tensor_network"},
				Description: "Complex interconnections",
			},

			// Process workloads
			"ideation": {
				Type:        TypeProcess,
				Category:    "process",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"context_mapping", "story_flow"},
				Description: "Idea generation",
			},
			"context_mapping": {
				Type:        TypeProcess,
				Category:    "process",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"ideation", "story_flow"},
				Description: "Context application",
			},
			"story_flow": {
				Type:        TypeProcess,
				Category:    "process",
				MinInLayer:  0,
				MaxInLayer:  1,
				ValidWith:   []string{"ideation", "context_mapping"},
				Description: "Narrative organization",
			},
		},
	}
	return v
}

type ValidationIssue struct {
	Level      string // "error" or "suggestion"
	Message    string
	Context    string
	Suggestion string
}

func (v *Validator) ValidateProcess(p Process) []ValidationIssue {
	var issues []ValidationIssue

	// Validate each layer
	for i, layer := range p.Layers {
		layerIssues := v.validateLayer(layer, i)
		issues = append(issues, layerIssues...)
	}

	// Validate overall process structure
	issues = append(issues, v.validateProcessStructure(p)...)

	return issues
}

func (v *Validator) validateLayer(layer Layer, index int) []ValidationIssue {
	var issues []ValidationIssue

	if len(layer.Workloads) == 0 {
		issues = append(issues, ValidationIssue{
			Level:      "error",
			Message:    fmt.Sprintf("Layer %d is empty", index+1),
			Suggestion: "Add at least one workload to the layer",
		})
		return issues
	}

	// Count workload types in this layer
	simCount := 0
	procCount := 0
	workloadNames := make(map[string]bool)

	for _, w := range layer.Workloads {
		if rule, exists := v.rules[w.Name]; exists {
			// Check for duplicates
			if workloadNames[w.Name] {
				issues = append(issues, ValidationIssue{
					Level:      "error",
					Message:    fmt.Sprintf("Duplicate workload '%s' in layer %d", w.Name, index+1),
					Suggestion: "Remove the duplicate workload",
				})
				continue
			}
			workloadNames[w.Name] = true

			if rule.Type == TypeSimulation {
				simCount++
			} else {
				procCount++
			}
		} else {
			issues = append(issues, ValidationIssue{
				Level:      "error",
				Message:    fmt.Sprintf("Unknown workload '%s' in layer %d", w.Name, index+1),
				Suggestion: "Use one of the defined workload types",
			})
		}
	}

	// Validate workload combinations
	if simCount > 0 && procCount > 0 {
		issues = append(issues, ValidationIssue{
			Level:      "suggestion",
			Message:    fmt.Sprintf("Layer %d mixes simulation and process workloads", index+1),
			Context:    "Simulation and process workloads typically work better in separate layers",
			Suggestion: "Consider separating simulation and process workloads",
		})
	}

	// Check for meaningful combinations
	if simCount >= 3 {
		issues = append(issues, ValidationIssue{
			Level:      "suggestion",
			Message:    fmt.Sprintf("Layer %d might be too complex with %d simulation workloads", index+1, simCount),
			Context:    "Multiple simulation workloads increase computational complexity",
			Suggestion: "Consider distributing simulation workloads across layers",
		})
	}

	return issues
}

func (v *Validator) validateProcessStructure(p Process) []ValidationIssue {
	var issues []ValidationIssue

	hasSimulation := false
	hasProcess := false

	// Check for presence of both simulation and process workloads
	for _, layer := range p.Layers {
		for _, w := range layer.Workloads {
			if rule, exists := v.rules[w.Name]; exists {
				if rule.Type == TypeSimulation {
					hasSimulation = true
				} else {
					hasProcess = true
				}
			}
		}
	}

	if !hasSimulation {
		issues = append(issues, ValidationIssue{
			Level:      "suggestion",
			Message:    "Process lacks simulation workloads",
			Context:    "Simulation workloads help create rich conceptual spaces",
			Suggestion: "Consider adding simulation workloads for deeper analysis",
		})
	}

	if !hasProcess {
		issues = append(issues, ValidationIssue{
			Level:      "suggestion",
			Message:    "Process lacks concrete processing workloads",
			Context:    "Process workloads help ground abstract concepts",
			Suggestion: "Consider adding process workloads to generate concrete outputs",
		})
	}

	return issues
}
