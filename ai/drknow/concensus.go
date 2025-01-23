package drknow

import (
	"sync"

	"github.com/theapemachine/errnie"
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

/*
ConsensusRule defines how perspectives are evaluated for consensus
*/
type ConsensusRule struct {
	Name     string
	Weight   float64
	Evaluate func([]Perspective) (interface{}, float64)
}

/*
ConsensusConfig defines the configuration for a consensus space.
*/
type ConsensusConfig struct {
	CollapseThreshold float64
	MinPerspectives   int
	Rules             []ConsensusRule
}

/*
NewConsensusSpace creates a new ConsensusSpace instance with the specified ID and configuration.
It initializes all the necessary maps and sets default values for the consensus parameters.

Parameters:
  - id: Unique identifier for the consensus space
  - config: Configuration settings for consensus rules and thresholds
*/
func NewConsensusSpace(id string, config ConsensusConfig) *ConsensusSpace {
	return &ConsensusSpace{
		ID:                id,
		Perspectives:      make([]Perspective, 0),
		Dependencies:      make(map[string][]string),
		WaitGroup:         make(map[string][]string),
		collapseThreshold: config.CollapseThreshold,
		minPerspectives:   config.MinPerspectives,
		consensusRules:    config.Rules,
		uncertainty:       qpool.MaxUncertainty,
		methodRegistry:    make(map[string][]Verification),
		diversity:         make(map[string]float64),
	}
}

/*
AddPerspective adds a new perspective to the consensus space and updates the quantum state.
It calculates method diversity, adjusts confidence, and attempts to reach consensus if possible.

Parameters:
  - p: The perspective to add to the consensus space
*/
func (cs *ConsensusSpace) AddPerspective(p Perspective) {
	errnie.Info("Adding perspective: %s", p.ID)

	errnie.Info("About to acquire lock for perspective: %s", p.ID)

	// Store the perspective
	cs.Perspectives = append(cs.Perspectives, p)

	// Calculate method diversity
	diversity := cs.calculateMethodDiversity(p.Method)
	errnie.Info("Method diversity: %f", diversity)

	// Adjust confidence based on method quality
	adjustedConfidence := p.Confidence * diversity
	errnie.Info("Adjusted confidence: %f", adjustedConfidence)

	// Update quantum uncertainty in qpool
	cs.uncertainty = qpool.UncertaintyLevel(1.0 - diversity)
	errnie.Info("Quantum uncertainty: %f", cs.uncertainty)

	// Create and store wave function in qpool
	cs.waveFunction = qpool.NewWaveFunction(
		[]qpool.State{
			{Value: p.Content, Probability: adjustedConfidence},
		},
		cs.uncertainty,
		adjustedConfidence,
	)

	errnie.Info("Notifying observers")

	if cs.OnNewPerspective != nil {
		cs.OnNewPerspective(p)
	}

	// Proceed with consensus building
	cs.tryCollapse()
}

/*
AddDependency establishes dependencies between agents in the consensus space.
It updates both the Dependencies map and WaitGroup to track agent relationships.

Parameters:
  - agentID: The ID of the agent that has dependencies
  - dependsOn: List of agent IDs that this agent depends on
*/
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

/*
tryCollapse attempts to reach consensus by evaluating current perspectives.
It checks if minimum conditions are met and collapses the quantum state if
confidence threshold is exceeded.
*/
func (cs *ConsensusSpace) tryCollapse() {
	errnie.Info("Trying to collapse")

	if cs.isCollapsed || len(cs.Perspectives) < cs.minPerspectives {
		errnie.Info("Not collapsing: isCollapsed or minPerspectives not met")
		return
	}

	consensus, confidence := cs.evaluateConsensus()
	errnie.Info("Consensus: %v, Confidence: %f", consensus, confidence)

	// Check if confidence exceeds collapse threshold
	if confidence >= cs.collapseThreshold {
		errnie.Info("Collapsing")
		cs.collapse(consensus)
	} else {
		errnie.Info("Not collapsing: confidence does not exceed collapse threshold")
		// Update uncertainty based on confidence
		cs.uncertainty = qpool.UncertaintyLevel(1.0 - confidence)
	}
}

/*
evaluateConsensus applies all consensus rules to the current perspectives
and returns the best result along with its confidence score.

Returns:
  - interface{}: The consensus result
  - float64: The confidence score for the consensus
*/
func (cs *ConsensusSpace) evaluateConsensus() (interface{}, float64) {
	errnie.Info("Evaluating consensus")

	var totalConfidence float64
	var weightedResults []struct {
		result interface{}
		weight float64
	}

	// Apply each consensus rule
	for _, rule := range cs.consensusRules {
		errnie.Info("Applying rule: %s", rule.Name)
		result, confidence := rule.Evaluate(cs.Perspectives)

		errnie.Info("Result: %v, Confidence: %f", result, confidence)
		weightedResults = append(weightedResults, struct {
			result interface{}
			weight float64
		}{
			result: result,
			weight: confidence * rule.Weight,
		})
		totalConfidence += confidence * rule.Weight
	}

	errnie.Info("Total confidence: %f", totalConfidence)

	// Find the result with highest weighted confidence
	var bestResult interface{}
	var maxWeight float64

	errnie.Info("Weighted results: %v", weightedResults)

	for _, wr := range weightedResults {
		if wr.weight > maxWeight {
			maxWeight = wr.weight
			bestResult = wr.result
		}
	}

	normalizedConfidence := totalConfidence / float64(len(cs.consensusRules))
	return bestResult, normalizedConfidence
}

/*
collapse finalizes the consensus state, sets minimum uncertainty,
and notifies observers of the final consensus value.

Parameters:
  - consensus: The final consensus value to be set
*/
func (cs *ConsensusSpace) collapse(consensus interface{}) {
	errnie.Info("Collapsing")

	cs.isCollapsed = true
	cs.consensus = consensus
	cs.uncertainty = qpool.MinUncertainty

	errnie.Info("Is collapsed: %t", cs.isCollapsed)
	errnie.Info("Consensus: %v", cs.consensus)
	errnie.Info("Uncertainty: %f", cs.uncertainty)

	if cs.OnCollapse != nil {
		cs.OnCollapse(consensus)
	}
}

/*
notifyDependents checks for agents waiting on the completed agent
and notifies them if all their dependencies are met.

Parameters:
  - completedAgentID: ID of the agent that just completed its task
*/
func (cs *ConsensusSpace) notifyDependents(completedAgentID string) {
	// Check agents waiting on this one
	for _, waitingID := range cs.WaitGroup[completedAgentID] {
		errnie.Info("Notifying dependent: %s", waitingID)

		// Check if all dependencies for this waiting agent are met
		deps := cs.Dependencies[waitingID]
		allMet := true

		for _, dep := range deps {
			if !cs.HasPerspective(dep) {
				allMet = false
				break
			}
		}

		errnie.Info("All dependencies met: %t", allMet)

		if allMet {
			// Remove from wait group and notify
			cs.removeFromWaitGroup(completedAgentID, waitingID)
			// Agent can now proceed (implementation specific)
		}
	}
}

/*
HasPerspective checks if a perspective from a specific agent exists
in the consensus space.

Parameters:
  - agentID: The ID of the agent to check for

Returns:
  - bool: True if the agent has submitted a perspective, false otherwise
*/
func (cs *ConsensusSpace) HasPerspective(agentID string) bool {
	for _, p := range cs.Perspectives {
		if p.ID == agentID {
			errnie.Info("Has perspective: %s", agentID)
			return true
		}
	}
	return false
}

/*
removeFromWaitGroup removes a waiting agent from the wait group
after its dependency is satisfied.

Parameters:
  - completedID: ID of the completed dependency
  - waitingID: ID of the waiting agent to remove
*/
func (cs *ConsensusSpace) removeFromWaitGroup(completedID, waitingID string) {
	waiting := cs.WaitGroup[completedID]
	for i, id := range waiting {
		if id == waitingID {
			cs.WaitGroup[completedID] = append(waiting[:i], waiting[i+1:]...)
			errnie.Info("Removed from wait group: %s", waitingID)
			return
		}
	}
}

/*
calculateMethodDiversity calculates and caches the diversity score
for a given method based on the current method registry.

Parameters:
  - method: The method to calculate diversity for

Returns:
  - float64: A diversity score between 0 and 1
*/
func (cs *ConsensusSpace) calculateMethodDiversity(method string) float64 {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	if cs.diversity == nil {
		cs.diversity = make(map[string]float64)
	}

	// Return cached diversity if exists
	if score, exists := cs.diversity[method]; exists {
		errnie.Info("Cached diversity: %f", score)
		return score
	}

	// Calculate new diversity score (simple implementation)
	totalMethods := float64(len(cs.methodRegistry))
	if totalMethods == 0 {
		errnie.Info("No methods found, returning 1.0")
		return 1.0
	}

	// Store and return new diversity score
	cs.diversity[method] = 1.0 / totalMethods
	errnie.Info("New diversity score: %f", cs.diversity[method])
	return cs.diversity[method]
}
