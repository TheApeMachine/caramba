package drknow

import (
	"sync"

	"github.com/theapemachine/qpool"
)

/*
ConsensusSpace builds on to of the qpool structures to provide a quantum-like
space where multiple perspectives exist in superposition until consensus is
reached, by collapsing a wave function into a single state.
*/
type ConsensusSpace struct {
	mu sync.RWMutex

	// Core state
	ID           string
	Perspectives []Perspective
	Dependencies map[string][]string // AgentID -> Dependencies
	WaitGroup    map[string][]string // AgentID -> Waiting Agents

	// Quantum properties
	isCollapsed bool
	consensus   interface{}
	uncertainty qpool.UncertaintyLevel

	// Consensus rules
	collapseThreshold float64 // Confidence threshold for collapse
	minPerspectives   int     // Minimum perspectives needed
	consensusRules    []ConsensusRule

	// Event handling
	OnCollapse       func(interface{})
	OnNewPerspective func(Perspective)

	methodRegistry map[string][]Verification
	diversity      map[string]float64
	waveFunction   *qpool.WaveFunction // Store the current wave function
}

// ConsensusRule defines how perspectives are evaluated for consensus
type ConsensusRule struct {
	Name     string
	Weight   float64
	Evaluate func([]Perspective) (interface{}, float64)
}

type ConsensusConfig struct {
	CollapseThreshold float64
	MinPerspectives   int
	Rules             []ConsensusRule
}

func NewConsensusSpace(id string, config ConsensusConfig) *ConsensusSpace {
	return &ConsensusSpace{
		ID:                id,
		Dependencies:      make(map[string][]string),
		WaitGroup:         make(map[string][]string),
		collapseThreshold: config.CollapseThreshold,
		minPerspectives:   config.MinPerspectives,
		consensusRules:    config.Rules,
		uncertainty:       qpool.MaxUncertainty,
	}
}

func (cs *ConsensusSpace) AddPerspective(p Perspective) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	// Store the perspective
	cs.Perspectives = append(cs.Perspectives, p)

	// Calculate method diversity
	diversity := cs.calculateMethodDiversity(p.Method)

	// Adjust confidence based on method quality
	adjustedConfidence := p.Confidence * diversity

	// Update quantum uncertainty in qpool
	cs.uncertainty = qpool.UncertaintyLevel(1.0 - diversity)

	// Create and store wave function in qpool
	cs.waveFunction = qpool.NewWaveFunction(
		[]qpool.State{
			{Value: p.Content, Probability: adjustedConfidence},
		},
		cs.uncertainty,
		adjustedConfidence,
	)

	// Notify observers
	if cs.OnNewPerspective != nil {
		cs.OnNewPerspective(p)
	}

	// Proceed with consensus building
	cs.tryCollapse()
}

// AddDependency establishes a dependency between agents
func (cs *ConsensusSpace) AddDependency(agentID string, dependsOn []string) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	cs.Dependencies[agentID] = dependsOn

	// Add to wait group if dependencies aren't met
	for _, dep := range dependsOn {
		if !cs.HasPerspective(dep) {
			cs.WaitGroup[dep] = append(cs.WaitGroup[dep], agentID)
		}
	}
}

// tryCollapse attempts to collapse the perspectives into a consensus
func (cs *ConsensusSpace) tryCollapse() {
	if cs.isCollapsed || len(cs.Perspectives) < cs.minPerspectives {
		return
	}

	consensus, confidence := cs.evaluateConsensus()

	// Check if confidence exceeds collapse threshold
	if confidence >= cs.collapseThreshold {
		cs.collapse(consensus)
	} else {
		// Update uncertainty based on confidence
		cs.uncertainty = qpool.UncertaintyLevel(1.0 - confidence)
	}
}

// evaluateConsensus applies consensus rules to current perspectives
func (cs *ConsensusSpace) evaluateConsensus() (interface{}, float64) {
	var totalConfidence float64
	var weightedResults []struct {
		result interface{}
		weight float64
	}

	// Apply each consensus rule
	for _, rule := range cs.consensusRules {
		result, confidence := rule.Evaluate(cs.Perspectives)
		weightedResults = append(weightedResults, struct {
			result interface{}
			weight float64
		}{
			result: result,
			weight: confidence * rule.Weight,
		})
		totalConfidence += confidence * rule.Weight
	}

	// Find the result with highest weighted confidence
	var bestResult interface{}
	var maxWeight float64

	for _, wr := range weightedResults {
		if wr.weight > maxWeight {
			maxWeight = wr.weight
			bestResult = wr.result
		}
	}

	normalizedConfidence := totalConfidence / float64(len(cs.consensusRules))
	return bestResult, normalizedConfidence
}

// collapse finalizes the consensus and notifies observers
func (cs *ConsensusSpace) collapse(consensus interface{}) {
	cs.isCollapsed = true
	cs.consensus = consensus
	cs.uncertainty = qpool.MinUncertainty

	if cs.OnCollapse != nil {
		cs.OnCollapse(consensus)
	}
}

// notifyDependents checks and notifies agents waiting on this one
func (cs *ConsensusSpace) notifyDependents(completedAgentID string) {
	// Check agents waiting on this one
	for _, waitingID := range cs.WaitGroup[completedAgentID] {
		// Check if all dependencies for this waiting agent are met
		deps := cs.Dependencies[waitingID]
		allMet := true

		for _, dep := range deps {
			if !cs.HasPerspective(dep) {
				allMet = false
				break
			}
		}

		if allMet {
			// Remove from wait group and notify
			cs.removeFromWaitGroup(completedAgentID, waitingID)
			// Agent can now proceed (implementation specific)
		}
	}
}

// Helper functions
func (cs *ConsensusSpace) HasPerspective(agentID string) bool {
	for _, p := range cs.Perspectives {
		if p.ID == agentID {
			return true
		}
	}
	return false
}

func (cs *ConsensusSpace) removeFromWaitGroup(completedID, waitingID string) {
	waiting := cs.WaitGroup[completedID]
	for i, id := range waiting {
		if id == waitingID {
			cs.WaitGroup[completedID] = append(waiting[:i], waiting[i+1:]...)
			return
		}
	}
}

// calculateMethodDiversity returns a diversity score (0-1) for a given method
func (cs *ConsensusSpace) calculateMethodDiversity(method string) float64 {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	if cs.diversity == nil {
		cs.diversity = make(map[string]float64)
	}

	// Return cached diversity if exists
	if score, exists := cs.diversity[method]; exists {
		return score
	}

	// Calculate new diversity score (simple implementation)
	totalMethods := float64(len(cs.methodRegistry))
	if totalMethods == 0 {
		return 1.0
	}

	// Store and return new diversity score
	cs.diversity[method] = 1.0 / totalMethods
	return cs.diversity[method]
}
