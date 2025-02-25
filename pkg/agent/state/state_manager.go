package state

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Now returns the current time
func Now() time.Time {
	return time.Now()
}

// State represents the state of an agent
type State struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Status        string                 `json:"status"`
	CurrentTask   string                 `json:"current_task"`
	Variables     map[string]interface{} `json:"variables"`
	History       []HistoryEntry         `json:"history"`
	Created       time.Time              `json:"created"`
	LastModified  time.Time              `json:"last_modified"`
}

// HistoryEntry represents an entry in the agent's history
type HistoryEntry struct {
	Timestamp time.Time              `json:"timestamp"`
	Action    string                 `json:"action"`
	Details   map[string]interface{} `json:"details"`
}

// Manager defines the interface for the state manager
type Manager interface {
	// CreateState creates a new state
	CreateState(ctx context.Context, name string) (string, error)
	
	// GetState retrieves a state by ID
	GetState(ctx context.Context, id string) (*State, error)
	
	// UpdateState updates a state
	UpdateState(ctx context.Context, id string, updater func(*State) error) error
	
	// DeleteState deletes a state
	DeleteState(ctx context.Context, id string) error
	
	// ListStates lists all states
	ListStates(ctx context.Context) ([]*State, error)
}

// InMemoryStateManager implements the StateManager interface with an in-memory map
type InMemoryStateManager struct {
	states map[string]*State
	mu     sync.RWMutex
}

// NewInMemoryStateManager creates a new in-memory state manager
func NewInMemoryStateManager() *InMemoryStateManager {
	return &InMemoryStateManager{
		states: make(map[string]*State),
	}
}

// CreateState creates a new state
func (m *InMemoryStateManager) CreateState(ctx context.Context, name string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	id := fmt.Sprintf("state-%d", time.Now().UnixNano())
	now := time.Now()
	
	state := &State{
		ID:           id,
		Name:         name,
		Status:       "created",
		Variables:    make(map[string]interface{}),
		History:      make([]HistoryEntry, 0),
		Created:      now,
		LastModified: now,
	}
	
	m.states[id] = state
	return id, nil
}

// GetState retrieves a state by ID
func (m *InMemoryStateManager) GetState(ctx context.Context, id string) (*State, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	state, exists := m.states[id]
	if !exists {
		return nil, fmt.Errorf("state not found: %s", id)
	}
	
	// Return a copy to prevent concurrent modification
	stateCopy := *state
	return &stateCopy, nil
}

// UpdateState updates a state
func (m *InMemoryStateManager) UpdateState(ctx context.Context, id string, updater func(*State) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	state, exists := m.states[id]
	if !exists {
		return fmt.Errorf("state not found: %s", id)
	}
	
	if err := updater(state); err != nil {
		return err
	}
	
	state.LastModified = time.Now()
	return nil
}

// DeleteState deletes a state
func (m *InMemoryStateManager) DeleteState(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if _, exists := m.states[id]; !exists {
		return fmt.Errorf("state not found: %s", id)
	}
	
	delete(m.states, id)
	return nil
}

// ListStates lists all states
func (m *InMemoryStateManager) ListStates(ctx context.Context) ([]*State, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	states := make([]*State, 0, len(m.states))
	for _, state := range m.states {
		// Return copies to prevent concurrent modification
		stateCopy := *state
		states = append(states, &stateCopy)
	}
	
	return states, nil
}

// SerializeState serializes a state to JSON
func SerializeState(state *State) ([]byte, error) {
	return json.Marshal(state)
}

// DeserializeState deserializes a state from JSON
func DeserializeState(data []byte) (*State, error) {
	var state State
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, err
	}
	return &state, nil
}
