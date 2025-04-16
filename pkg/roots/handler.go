package roots

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// MCPHandler handles MCP requests for roots
type MCPHandler struct {
	manager RootsManager
}

// NewMCPHandler creates a new MCPHandler
func NewMCPHandler(manager RootsManager) *MCPHandler {
	return &MCPHandler{
		manager: manager,
	}
}

// HandleListRoots handles the roots/list request
func (h *MCPHandler) HandleListRoots(ctx context.Context, params json.RawMessage) (*mcp.ListRootsResult, error) {
	roots, err := h.manager.List(ctx)
	if err != nil {
		return nil, errnie.New(errnie.WithError(
			&ListRootsError{},
		))
	}

	// Convert roots to MCP format
	mcpRoots := make([]mcp.Root, 0, len(roots))
	for _, root := range roots {
		mcpRoot := mcp.Root{
			URI:  root.URI,
			Name: root.Name,
		}
		mcpRoots = append(mcpRoots, mcpRoot)
	}

	return &mcp.ListRootsResult{
		Roots: mcpRoots,
	}, nil
}

// HandleGetRoot handles the roots/get request
func (h *MCPHandler) HandleGetRoot(ctx context.Context, params json.RawMessage) (*mcp.Root, error) {
	// Parse request parameters
	var req struct {
		URI string `json:"uri"`
	}
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, errnie.New(errnie.WithError(
			&ParseRequestError{RootURI: req.URI},
		))
	}

	// Find the root with the matching URI
	roots, err := h.manager.List(ctx)
	if err != nil {
		return nil, errnie.New(errnie.WithError(
			&ListRootsError{},
		))
	}

	for _, root := range roots {
		if root.URI == req.URI {
			// Convert to MCP format
			return &mcp.Root{
				URI:  root.URI,
				Name: root.Name,
			}, nil
		}
	}

	return nil, errnie.New(errnie.WithError(
		&RootNotFoundError{RootURI: req.URI},
	))
}

// HandleCreateRoot handles the roots/create request
func (h *MCPHandler) HandleCreateRoot(ctx context.Context, params json.RawMessage) (*mcp.Root, error) {
	// Parse request parameters
	var req struct {
		URI  string `json:"uri"`
		Name string `json:"name,omitempty"`
	}
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, errnie.New(errnie.WithError(
			&ParseRequestError{RootURI: req.URI},
		))
	}

	// Create the root
	root := Root{
		URI:  req.URI,
		Name: req.Name,
	}

	createdRoot, err := h.manager.Create(ctx, root)
	if err != nil {
		return nil, errnie.New(errnie.WithError(
			&RootCreateError{RootURI: req.URI},
		))
	}

	// Convert back to MCP format
	return &mcp.Root{
		URI:  createdRoot.URI,
		Name: createdRoot.Name,
	}, nil
}

// HandleUpdateRoot handles the roots/update request
func (h *MCPHandler) HandleUpdateRoot(ctx context.Context, params json.RawMessage) (*mcp.Root, error) {
	// Parse request parameters
	var req struct {
		URI  string `json:"uri"`
		Name string `json:"name,omitempty"`
	}
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, errnie.New(errnie.WithError(
			&ParseRequestError{RootURI: req.URI},
		))
	}

	// Find the root with the matching URI
	roots, err := h.manager.List(ctx)
	if err != nil {
		return nil, errnie.New(errnie.WithError(
			&ListRootsError{},
		))
	}

	var existingRoot *Root
	for _, root := range roots {
		if root.URI == req.URI {
			existingRoot = &root
			break
		}
	}

	if existingRoot == nil {
		return nil, errnie.New(errnie.WithError(
			&RootNotFoundError{RootURI: req.URI},
		))
	}

	// Update the root
	existingRoot.Name = req.Name
	updatedRoot, err := h.manager.Update(ctx, *existingRoot)
	if err != nil {
		return nil, errnie.New(errnie.WithError(
			&RootUpdateError{RootURI: req.URI},
		))
	}

	// Convert back to MCP format
	return &mcp.Root{
		URI:  updatedRoot.URI,
		Name: updatedRoot.Name,
	}, nil
}

// HandleDeleteRoot handles the roots/delete request
func (h *MCPHandler) HandleDeleteRoot(ctx context.Context, params json.RawMessage) (bool, error) {
	// Parse request parameters
	var req struct {
		URI string `json:"uri"`
	}
	if err := json.Unmarshal(params, &req); err != nil {
		return false, errnie.New(errnie.WithError(
			&ParseRequestError{RootURI: req.URI},
		))
	}

	// Find the root with the matching URI
	roots, err := h.manager.List(ctx)
	if err != nil {
		return false, errnie.New(errnie.WithError(
			&ListRootsError{},
		))
	}

	var rootID string
	for _, root := range roots {
		if root.URI == req.URI {
			rootID = root.ID
			break
		}
	}

	if rootID == "" {
		return false, fmt.Errorf("root not found: %s", req.URI)
	}

	// Delete the root
	err = h.manager.Delete(ctx, rootID)
	if err != nil {
		return false, errnie.New(errnie.WithError(
			&RootDeleteError{RootURI: req.URI},
		))
	}

	return true, nil
}

// HandleSubscribeToRoots handles the roots/subscribe request
func (h *MCPHandler) HandleSubscribeToRoots(ctx context.Context, params json.RawMessage) (string, error) {
	// Subscribe to root changes
	ch, err := h.manager.Subscribe(ctx)
	if err != nil {
		return "", errnie.New(errnie.WithError(
			&RootSubscribeError{},
		))
	}

	// Generate a subscription ID
	subscriptionID := fmt.Sprintf("roots-%d", len(ch))

	// Start a goroutine to handle notifications
	go func() {
		for change := range ch {
			// TODO: Send notification to client
			// This would typically involve sending an SSE event or WebSocket message
			_ = change
		}
	}()

	return subscriptionID, nil
}

// HandleUnsubscribeFromRoots handles the roots/unsubscribe request
func (h *MCPHandler) HandleUnsubscribeFromRoots(ctx context.Context, params json.RawMessage) (bool, error) {
	// Parse request parameters
	var req struct {
		SubscriptionID string `json:"subscriptionId"`
	}
	if err := json.Unmarshal(params, &req); err != nil {
		return false, errnie.New(errnie.WithError(
			&ParseRequestError{SubscriptionID: req.SubscriptionID},
		))
	}

	// Unsubscribe from root changes
	err := h.manager.Unsubscribe(ctx, req.SubscriptionID)
	if err != nil {
		return false, errnie.New(errnie.WithError(
			&RootUnsubscribeError{SubscriptionID: req.SubscriptionID},
		))
	}

	return true, nil
}

type RootNotFoundError struct {
	RootURI string
}

func (e *RootNotFoundError) Error() string {
	return fmt.Sprintf("root not found: %s", e.RootURI)
}

func (e *RootNotFoundError) Is(target error) bool {
	return target == e
}

type RootAlreadyExistsError struct {
	RootURI string
}

func (e *RootAlreadyExistsError) Error() string {
	return fmt.Sprintf("root already exists: %s", e.RootURI)
}

func (e *RootAlreadyExistsError) Is(target error) bool {
	return target == e
}

type RootUpdateError struct {
	RootURI string
}

func (e *RootUpdateError) Error() string {
	return fmt.Sprintf("failed to update root: %s", e.RootURI)
}

func (e *RootUpdateError) Is(target error) bool {
	return target == e
}

type RootCreateError struct {
	RootURI string
}

func (e *RootCreateError) Error() string {
	return fmt.Sprintf("failed to create root: %s", e.RootURI)
}

func (e *RootCreateError) Is(target error) bool {
	return target == e
}

type RootDeleteError struct {
	RootURI string
}

func (e *RootDeleteError) Error() string {
	return fmt.Sprintf("failed to delete root: %s", e.RootURI)
}

func (e *RootDeleteError) Is(target error) bool {
	return target == e
}

type RootSubscribeError struct {
	RootURI string
}

func (e *RootSubscribeError) Error() string {
	return fmt.Sprintf("failed to subscribe to root: %s", e.RootURI)
}

func (e *RootSubscribeError) Is(target error) bool {
	return target == e
}

type RootUnsubscribeError struct {
	SubscriptionID string
}

func (e *RootUnsubscribeError) Error() string {
	return fmt.Sprintf("failed to unsubscribe from root: %s", e.SubscriptionID)
}

func (e *RootUnsubscribeError) Is(target error) bool {
	return target == e
}

type ParseRequestError struct {
	RootURI        string
	SubscriptionID string
}

func (e *ParseRequestError) Error() string {
	return fmt.Sprintf("failed to parse request: %s", e.RootURI)
}

func (e *ParseRequestError) Is(target error) bool {
	return target == e
}

type ListRootsError struct {
	RootURI string
}

func (e *ListRootsError) Error() string {
	return fmt.Sprintf("failed to list roots: %s", e.RootURI)
}

func (e *ListRootsError) Is(target error) bool {
	return target == e
}
