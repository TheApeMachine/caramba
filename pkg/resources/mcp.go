package resources

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

// MCPHandler handles MCP resource requests
type MCPHandler struct {
	manager ResourceManager
}

// NewMCPHandler creates a new MCPHandler instance
func NewMCPHandler(manager ResourceManager) *MCPHandler {
	return &MCPHandler{
		manager: manager,
	}
}

// HandleListResources handles the resources/list request
func (h *MCPHandler) HandleListResources(ctx context.Context, req *mcp.ListResourcesRequest) (*mcp.ListResourcesResult, error) {
	resources, templates, err := h.manager.List(ctx)
	if err != nil {
		return nil, err
	}

	// Convert resources to MCP format
	mcpResources := make([]mcp.Resource, len(resources))
	for i, r := range resources {
		mcpResources[i] = mcp.NewResource(r.URI, r.Name,
			mcp.WithResourceDescription(r.Description),
			mcp.WithMIMEType(r.MimeType),
		)
	}

	// Convert templates to MCP format
	mcpTemplates := make([]mcp.ResourceTemplate, len(templates))
	for i, t := range templates {
		mcpTemplates[i] = mcp.NewResourceTemplate(t.URITemplate, t.Name,
			mcp.WithTemplateDescription(t.Description),
			mcp.WithTemplateMIMEType(t.MimeType),
		)
	}

	// Create result with templates in meta field
	result := &mcp.ListResourcesResult{
		Resources: mcpResources,
	}

	// Add templates to the result's meta field
	if len(mcpTemplates) > 0 {
		result.Meta = map[string]interface{}{
			"templates": mcpTemplates,
		}
	}

	return result, nil
}

// HandleReadResource handles the resources/read request
func (h *MCPHandler) HandleReadResource(ctx context.Context, req *mcp.ReadResourceRequest) (*mcp.ReadResourceResult, error) {
	contents, err := h.manager.Read(ctx, req.Params.URI)
	if err != nil {
		return nil, err
	}

	// Convert contents to MCP format
	mcpContents := make([]mcp.ResourceContents, len(contents))
	for i, c := range contents {
		if c.Text != "" {
			mcpContents[i] = &mcp.TextResourceContents{
				URI:      c.URI,
				MIMEType: c.MimeType,
				Text:     c.Text,
			}
		} else {
			mcpContents[i] = &mcp.BlobResourceContents{
				URI:      c.URI,
				MIMEType: c.MimeType,
				Blob:     c.Blob,
			}
		}
	}

	return &mcp.ReadResourceResult{
		Contents: mcpContents,
	}, nil
}
