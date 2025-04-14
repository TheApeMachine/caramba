package roots

import (
	"context"
	"time"
)

// Root represents a root URI in the system
type Root struct {
	ID          string    `json:"id"`
	URI         string    `json:"uri"`
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	CreatedAt   time.Time `json:"createdAt"`
	UpdatedAt   time.Time `json:"updatedAt"`
	Metadata    any       `json:"metadata,omitempty"`
}

// RootChange represents a change to a root
type RootChange struct {
	ID        string     `json:"id"`
	RootID    string     `json:"rootId"`
	Type      ChangeType `json:"type"`
	Timestamp time.Time  `json:"timestamp"`
	Metadata  any        `json:"metadata,omitempty"`
}

// ChangeType represents the type of change to a root
type ChangeType string

const (
	// RootAdded indicates a root was added
	RootAdded ChangeType = "added"
	// RootRemoved indicates a root was removed
	RootRemoved ChangeType = "removed"
	// RootUpdated indicates a root was updated
	RootUpdated ChangeType = "updated"
)

// RootsManager defines the interface for managing roots
type RootsManager interface {
	// List returns all available roots
	List(ctx context.Context) ([]Root, error)

	// Get retrieves a root by ID
	Get(ctx context.Context, id string) (*Root, error)

	// Create creates a new root
	Create(ctx context.Context, root Root) (*Root, error)

	// Update updates an existing root
	Update(ctx context.Context, root Root) (*Root, error)

	// Delete deletes a root
	Delete(ctx context.Context, id string) error

	// Subscribe subscribes to root changes
	Subscribe(ctx context.Context) (<-chan *RootChange, error)

	// Unsubscribe unsubscribes from root changes
	Unsubscribe(ctx context.Context, id string) error
}
