package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools/trengo"
)

// TrengoTool provides a base for all Trengo operations
type TrengoTool struct {
	*ToolBuilder
	pctx   context.Context
	ctx    context.Context
	cancel context.CancelFunc
	client *trengo.Client
}

type TrengoToolOption func(*TrengoTool)

// NewTrengoTool creates a new Trengo tool with options
func NewTrengoTool(opts ...TrengoToolOption) *TrengoTool {
	ctx, cancel := context.WithCancel(context.Background())

	client := trengo.NewClient()

	tool := &TrengoTool{
		ToolBuilder: NewToolBuilder(),
		ctx:         ctx,
		cancel:      cancel,
		client:      client,
	}

	for _, opt := range opts {
		opt(tool)
	}

	return tool
}

func WithTrengoCancel(ctx context.Context) TrengoToolOption {
	return func(tool *TrengoTool) {
		tool.pctx = ctx
	}
}

func (tool *TrengoTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("trengo.TrengoTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-tool.pctx.Done():
				errnie.Debug("trengo.TrengoTool.Generate: parent context done")
				tool.cancel()
				return
			case <-tool.ctx.Done():
				errnie.Debug("trengo.TrengoTool.Generate: context done")
				return
			case artifact := <-buffer:
				for _, f := range fn {
					out <- f(artifact)
				}
			}
		}
	}()

	return out
}

// TrengoListTicketsTool implements a tool for listing tickets
type TrengoListTicketsTool struct {
	*TrengoTool
}

func NewTrengoListTicketsTool() *TrengoListTicketsTool {
	// Create MCP tool definition based on schema from config.yml
	listTicketsTool := mcp.NewTool(
		"list_tickets",
		mcp.WithDescription("A tool for listing tickets in Trengo."),
	)

	tltt := &TrengoListTicketsTool{
		TrengoTool: NewTrengoTool(),
	}

	tltt.ToolBuilder.mcp = &listTicketsTool
	return tltt
}

func (tool *TrengoListTicketsTool) ID() string {
	return "trengo_list_tickets"
}

func (tool *TrengoListTicketsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoListTicketsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing tickets
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoListTicketsTool
func (tool *TrengoListTicketsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoCreateTicketTool implements a tool for creating tickets
type TrengoCreateTicketTool struct {
	*TrengoTool
}

func NewTrengoCreateTicketTool() *TrengoCreateTicketTool {
	// Create MCP tool definition based on schema from config.yml
	createTicketTool := mcp.NewTool(
		"create_ticket",
		mcp.WithDescription("A tool for creating tickets in Trengo."),
	)

	tctt := &TrengoCreateTicketTool{
		TrengoTool: NewTrengoTool(),
	}

	tctt.ToolBuilder.mcp = &createTicketTool
	return tctt
}

func (tool *TrengoCreateTicketTool) ID() string {
	return "trengo_create_ticket"
}

func (tool *TrengoCreateTicketTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoCreateTicketTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating tickets
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoCreateTicketTool
func (tool *TrengoCreateTicketTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoAssignTicketTool implements a tool for assigning tickets
type TrengoAssignTicketTool struct {
	*TrengoTool
}

func NewTrengoAssignTicketTool() *TrengoAssignTicketTool {
	// Create MCP tool definition based on schema from config.yml
	assignTicketTool := mcp.NewTool(
		"assign_ticket",
		mcp.WithDescription("A tool for assigning tickets in Trengo."),
	)

	tatt := &TrengoAssignTicketTool{
		TrengoTool: NewTrengoTool(),
	}

	tatt.ToolBuilder.mcp = &assignTicketTool
	return tatt
}

func (tool *TrengoAssignTicketTool) ID() string {
	return "trengo_assign_ticket"
}

func (tool *TrengoAssignTicketTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoAssignTicketTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for assigning tickets
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoAssignTicketTool
func (tool *TrengoAssignTicketTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoCloseTicketTool implements a tool for closing tickets
type TrengoCloseTicketTool struct {
	*TrengoTool
}

func NewTrengoCloseTicketTool() *TrengoCloseTicketTool {
	// Create MCP tool definition based on schema from config.yml
	closeTicketTool := mcp.NewTool(
		"close_ticket",
		mcp.WithDescription("A tool for closing tickets in Trengo."),
	)

	tctt := &TrengoCloseTicketTool{
		TrengoTool: NewTrengoTool(),
	}

	tctt.ToolBuilder.mcp = &closeTicketTool
	return tctt
}

func (tool *TrengoCloseTicketTool) ID() string {
	return "trengo_close_ticket"
}

func (tool *TrengoCloseTicketTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoCloseTicketTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for closing tickets
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoCloseTicketTool
func (tool *TrengoCloseTicketTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoReopenTicketTool implements a tool for reopening tickets
type TrengoReopenTicketTool struct {
	*TrengoTool
}

func NewTrengoReopenTicketTool() *TrengoReopenTicketTool {
	// Create MCP tool definition based on schema from config.yml
	reopenTicketTool := mcp.NewTool(
		"reopen_ticket",
		mcp.WithDescription("A tool for reopening tickets in Trengo."),
	)

	trtt := &TrengoReopenTicketTool{
		TrengoTool: NewTrengoTool(),
	}

	trtt.ToolBuilder.mcp = &reopenTicketTool
	return trtt
}

func (tool *TrengoReopenTicketTool) ID() string {
	return "trengo_reopen_ticket"
}

func (tool *TrengoReopenTicketTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoReopenTicketTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for reopening tickets
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoReopenTicketTool
func (tool *TrengoReopenTicketTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoListLabelsTool implements a tool for listing labels
type TrengoListLabelsTool struct {
	*TrengoTool
}

func NewTrengoListLabelsTool() *TrengoListLabelsTool {
	// Create MCP tool definition based on schema from config.yml
	listLabelsTool := mcp.NewTool(
		"list_labels",
		mcp.WithDescription("A tool for listing labels in Trengo."),
	)

	tllt := &TrengoListLabelsTool{
		TrengoTool: NewTrengoTool(),
	}

	tllt.ToolBuilder.mcp = &listLabelsTool
	return tllt
}

func (tool *TrengoListLabelsTool) ID() string {
	return "trengo_list_labels"
}

func (tool *TrengoListLabelsTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoListLabelsTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for listing labels
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoListLabelsTool
func (tool *TrengoListLabelsTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoGetLabelTool implements a tool for getting labels
type TrengoGetLabelTool struct {
	*TrengoTool
}

func NewTrengoGetLabelTool() *TrengoGetLabelTool {
	// Create MCP tool definition based on schema from config.yml
	getLabelTool := mcp.NewTool(
		"get_label",
		mcp.WithDescription("A tool for getting a label in Trengo."),
	)

	tglt := &TrengoGetLabelTool{
		TrengoTool: NewTrengoTool(),
	}

	tglt.ToolBuilder.mcp = &getLabelTool
	return tglt
}

func (tool *TrengoGetLabelTool) ID() string {
	return "trengo_get_label"
}

func (tool *TrengoGetLabelTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoGetLabelTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for getting a label
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoGetLabelTool
func (tool *TrengoGetLabelTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoCreateLabelTool implements a tool for creating labels
type TrengoCreateLabelTool struct {
	*TrengoTool
}

func NewTrengoCreateLabelTool() *TrengoCreateLabelTool {
	// Create MCP tool definition based on schema from config.yml
	createLabelTool := mcp.NewTool(
		"create_label",
		mcp.WithDescription("A tool for creating a label in Trengo."),
	)

	tclt := &TrengoCreateLabelTool{
		TrengoTool: NewTrengoTool(),
	}

	tclt.ToolBuilder.mcp = &createLabelTool
	return tclt
}

func (tool *TrengoCreateLabelTool) ID() string {
	return "trengo_create_label"
}

func (tool *TrengoCreateLabelTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoCreateLabelTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for creating a label
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoCreateLabelTool
func (tool *TrengoCreateLabelTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoUpdateLabelTool implements a tool for updating labels
type TrengoUpdateLabelTool struct {
	*TrengoTool
}

func NewTrengoUpdateLabelTool() *TrengoUpdateLabelTool {
	// Create MCP tool definition based on schema from config.yml
	updateLabelTool := mcp.NewTool(
		"update_label",
		mcp.WithDescription("A tool for updating a label in Trengo."),
	)

	tult := &TrengoUpdateLabelTool{
		TrengoTool: NewTrengoTool(),
	}

	tult.ToolBuilder.mcp = &updateLabelTool
	return tult
}

func (tool *TrengoUpdateLabelTool) ID() string {
	return "trengo_update_label"
}

func (tool *TrengoUpdateLabelTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoUpdateLabelTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for updating a label
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoUpdateLabelTool
func (tool *TrengoUpdateLabelTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// TrengoDeleteLabelTool implements a tool for deleting labels
type TrengoDeleteLabelTool struct {
	*TrengoTool
}

func NewTrengoDeleteLabelTool() *TrengoDeleteLabelTool {
	// Create MCP tool definition based on schema from config.yml
	deleteLabelTool := mcp.NewTool(
		"delete_label",
		mcp.WithDescription("A tool for deleting a label in Trengo."),
	)

	tdlt := &TrengoDeleteLabelTool{
		TrengoTool: NewTrengoTool(),
	}

	tdlt.ToolBuilder.mcp = &deleteLabelTool
	return tdlt
}

func (tool *TrengoDeleteLabelTool) ID() string {
	return "trengo_delete_label"
}

func (tool *TrengoDeleteLabelTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.TrengoTool.Generate(buffer, tool.fn)
}

func (tool *TrengoDeleteLabelTool) fn(artifact *datura.Artifact) *datura.Artifact {
	// Implementation for deleting a label
	return artifact
}

// ToMCP returns the MCP tool definitions for the TrengoDeleteLabelTool
func (tool *TrengoDeleteLabelTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}
