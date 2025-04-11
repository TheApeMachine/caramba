package azure

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Client provides a high-level interface to Azure DevOps services.
It manages connections and operations for both work items and wiki pages
through a unified streaming interface.
*/
type Client struct {
	conn     *azuredevops.Connection
	workitem *WorkItem
	wiki     *Wiki
}

/*
NewClient creates a new Azure DevOps client using environment variables for authentication.

It initializes connections to work item and wiki services using a personal access token.
The client uses AZDO_ORG_URL and AZDO_PAT environment variables.
*/
func NewClient() *Client {
	conn := azuredevops.NewPatConnection(
		os.Getenv("AZDO_ORG_URL"),
		os.Getenv("AZDO_PAT"),
	)

	workitem := NewWorkItem(conn)
	wiki := NewWiki(conn)

	return &Client{
		conn:     conn,
		workitem: workitem,
		wiki:     wiki,
	}
}

/*
Do handles incoming tool calls for Azure DevOps operations.
It routes the request to the appropriate service (WorkItem or Wiki)
and executes the specified operation.
*/
func (c *Client) Do(toolcall mcp.CallToolRequest) mcp.CallToolResult {
	ctx := context.Background()
	args := toolcall.Params.Arguments
	parts := strings.Split(toolcall.Params.Name, ".")

	if len(parts) != 3 {
		errMsg := fmt.Sprintf("invalid operation format: expected 'azure.<service>.<action>', got '%s'", toolcall.Params.Name)
		errnie.Error(fmt.Errorf(errMsg))
		return mcp.CallToolResult{
			Content: []mcp.Content{mcp.TextContent{Type: "text", Text: errMsg}},
		}
	}

	service := parts[1]
	action := parts[2]

	var result interface{}
	var err error

	switch service {
	case "workitem":
		if c.workitem == nil {
			errMsg := "workitem client not initialized"
			errnie.Error(fmt.Errorf(errMsg))
			return mcp.CallToolResult{
				Content: []mcp.Content{mcp.TextContent{Type: "text", Text: errMsg}},
			}
		}
		switch action {
		case "create":
			result, err = c.workitem.CreateWorkItem(ctx, args)
		case "update":
			result, err = c.workitem.UpdateWorkItem(ctx, args)
		case "get":
			result, err = c.workitem.GetWorkItem(ctx, args)
		case "list":
			result, err = c.workitem.ListWorkItems(ctx, args)
		default:
			err = fmt.Errorf("unknown workitem action: %s", action)
		}
	case "wiki":
		if c.wiki == nil {
			errMsg := "wiki client not initialized"
			errnie.Error(fmt.Errorf(errMsg))
			return mcp.CallToolResult{
				Content: []mcp.Content{mcp.TextContent{Type: "text", Text: errMsg}},
			}
		}
		switch action {
		case "create_page":
			result, err = c.wiki.CreatePage(ctx, args)
		case "update_page":
			result, err = c.wiki.UpdatePage(ctx, args)
		case "get_page":
			result, err = c.wiki.GetPage(ctx, args)
		case "list_pages":
			result, err = c.wiki.ListPages(ctx, args)
		default:
			err = fmt.Errorf("unknown wiki action: %s", action)
		}
	default:
		err = fmt.Errorf("unknown azure service: %s", service)
	}

	if err != nil {
		errnie.Error(err)
		return mcp.CallToolResult{
			Content: []mcp.Content{mcp.TextContent{Type: "text", Text: err.Error()}},
		}
	}

	// Marshal the result to JSON
	jsonResult, marshalErr := json.MarshalIndent(result, "", "  ")
	if marshalErr != nil {
		errnie.Error(marshalErr)
		return mcp.CallToolResult{
			Content: []mcp.Content{mcp.TextContent{Type: "text", Text: fmt.Sprintf("failed to marshal result: %s", marshalErr.Error())}},
		}
	}

	return mcp.CallToolResult{
		Content: []mcp.Content{mcp.TextContent{Type: "text", Text: string(jsonResult)}},
	}
}
