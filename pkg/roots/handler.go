package roots

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
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
		return nil, fmt.Errorf("failed to list roots: %w", err)
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
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}

	// Find the root with the matching URI
	roots, err := h.manager.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list roots: %w", err)
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

	return nil, fmt.Errorf("root not found: %s", req.URI)
}

// HandleCreateRoot handles the roots/create request
func (h *MCPHandler) HandleCreateRoot(ctx context.Context, params json.RawMessage) (*mcp.Root, error) {
	// Parse request parameters
	var req struct {
		URI  string `json:"uri"`
		Name string `json:"name,omitempty"`
	}
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}

	// Create the root
	root := Root{
		URI:  req.URI,
		Name: req.Name,
	}

	createdRoot, err := h.manager.Create(ctx, root)
	if err != nil {
		return nil, fmt.Errorf("failed to create root: %w", err)
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
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}

	// Find the root with the matching URI
	roots, err := h.manager.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list roots: %w", err)
	}

	var existingRoot *Root
	for _, root := range roots {
		if root.URI == req.URI {
			existingRoot = &root
			break
		}
	}

	if existingRoot == nil {
		return nil, fmt.Errorf("root not found: %s", req.URI)
	}

	// Update the root
	existingRoot.Name = req.Name
	updatedRoot, err := h.manager.Update(ctx, *existingRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to update root: %w", err)
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
		return false, fmt.Errorf("failed to parse request: %w", err)
	}

	// Find the root with the matching URI
	roots, err := h.manager.List(ctx)
	if err != nil {
		return false, fmt.Errorf("failed to list roots: %w", err)
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
		return false, fmt.Errorf("failed to delete root: %w", err)
	}

	return true, nil
}

// HandleSubscribeToRoots handles the roots/subscribe request
func (h *MCPHandler) HandleSubscribeToRoots(ctx context.Context, params json.RawMessage) (string, error) {
	// Subscribe to root changes
	ch, err := h.manager.Subscribe(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to subscribe to roots: %w", err)
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
		return false, fmt.Errorf("failed to parse request: %w", err)
	}

	// Unsubscribe from root changes
	err := h.manager.Unsubscribe(ctx, req.SubscriptionID)
	if err != nil {
		return false, fmt.Errorf("failed to unsubscribe from roots: %w", err)
	}

	return true, nil
}
