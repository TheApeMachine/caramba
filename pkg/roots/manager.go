package roots

import (
	"context"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// DefaultManager is the default implementation of RootsManager
type DefaultManager struct {
	roots       map[string]*Root
	subscribers map[string]chan *RootChange
	mu          sync.RWMutex
}

// NewDefaultManager creates a new DefaultManager instance
func NewDefaultManager() *DefaultManager {
	return &DefaultManager{
		roots:       make(map[string]*Root),
		subscribers: make(map[string]chan *RootChange),
	}
}

// List returns all available roots
func (m *DefaultManager) List(ctx context.Context) ([]Root, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	roots := make([]Root, 0, len(m.roots))
	for _, root := range m.roots {
		roots = append(roots, *root)
	}

	return roots, nil
}

// Get retrieves a root by ID
func (m *DefaultManager) Get(ctx context.Context, id string) (*Root, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	root, ok := m.roots[id]
	if !ok {
		return nil, errnie.New(errnie.WithError(
			&RootNotFoundError{RootURI: id},
		))
	}

	return root, nil
}

// Create creates a new root
func (m *DefaultManager) Create(ctx context.Context, root Root) (*Root, error) {
	// Generate a new ID if not provided
	if root.ID == "" {
		root.ID = uuid.New().String()
	}

	// Set timestamps
	now := time.Now()
	root.CreatedAt = now
	root.UpdatedAt = now

	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if root with same URI already exists
	for _, existing := range m.roots {
		if existing.URI == root.URI {
			return nil, errnie.New(errnie.WithError(
				&RootAlreadyExistsError{RootURI: root.URI},
			))
		}
	}

	// Store the root
	m.roots[root.ID] = &root

	// Notify subscribers
	change := &RootChange{
		ID:        uuid.New().String(),
		RootID:    root.ID,
		Type:      RootAdded,
		Timestamp: now,
	}
	m.notifySubscribers(change)

	return &root, nil
}

// Update updates an existing root
func (m *DefaultManager) Update(ctx context.Context, root Root) (*Root, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if the root exists
	existingRoot, ok := m.roots[root.ID]
	if !ok {
		return nil, errnie.New(errnie.WithError(
			&RootNotFoundError{RootURI: root.ID},
		))
	}

	// Check if URI is being changed and if it conflicts with another root
	if root.URI != existingRoot.URI {
		for _, other := range m.roots {
			if other.ID != root.ID && other.URI == root.URI {
				return nil, errnie.New(errnie.WithError(
					&RootAlreadyExistsError{RootURI: root.URI},
				))
			}
		}
	}

	// Update the root
	root.CreatedAt = existingRoot.CreatedAt
	root.UpdatedAt = time.Now()
	m.roots[root.ID] = &root

	// Notify subscribers
	change := &RootChange{
		ID:        uuid.New().String(),
		RootID:    root.ID,
		Type:      RootUpdated,
		Timestamp: root.UpdatedAt,
	}
	m.notifySubscribers(change)

	return &root, nil
}

// Delete deletes a root
func (m *DefaultManager) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if the root exists
	if _, ok := m.roots[id]; !ok {
		return errnie.New(errnie.WithError(
			&RootNotFoundError{RootURI: id},
		))
	}

	// Delete the root
	delete(m.roots, id)

	// Notify subscribers
	change := &RootChange{
		ID:        uuid.New().String(),
		RootID:    id,
		Type:      RootRemoved,
		Timestamp: time.Now(),
	}
	m.notifySubscribers(change)

	return nil
}

// Subscribe subscribes to root changes
func (m *DefaultManager) Subscribe(ctx context.Context) (<-chan *RootChange, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Create a new subscriber channel
	subID := uuid.New().String()
	ch := make(chan *RootChange, 10)
	m.subscribers[subID] = ch

	// Start a goroutine to clean up the subscription when the context is done
	go func() {
		<-ctx.Done()
		m.Unsubscribe(ctx, subID)
	}()

	return ch, nil
}

// Unsubscribe unsubscribes from root changes
func (m *DefaultManager) Unsubscribe(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	ch, ok := m.subscribers[id]
	if !ok {
		return errnie.New(errnie.WithError(
			&RootUnsubscribeError{SubscriptionID: id},
		))
	}

	// Close the channel and remove the subscriber
	close(ch)
	delete(m.subscribers, id)

	return nil
}

// notifySubscribers notifies all subscribers of a root change
func (m *DefaultManager) notifySubscribers(change *RootChange) {
	for _, ch := range m.subscribers {
		select {
		case ch <- change:
			// Message sent successfully
		default:
			// Channel is full, skip this notification
		}
	}
}
