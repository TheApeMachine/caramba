package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools/editor"
)

// EditorTool provides common functionality for all editor tools
type EditorTool struct {
	*ToolBuilder
	pctx   context.Context
	ctx    context.Context
	cancel context.CancelFunc
	client *editor.Client
}

type EditorToolOption func(*EditorTool)

// NewEditorTool creates a new editor tool with the specified options
func NewEditorTool(opts ...EditorToolOption) *EditorTool {
	ctx, cancel := context.WithCancel(context.Background())

	client := editor.NewClient()

	tool := &EditorTool{
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

// WithEditorCancel sets the parent context for an editor tool
func WithEditorCancel(ctx context.Context) EditorToolOption {
	return func(tool *EditorTool) {
		tool.pctx = ctx
	}
}

// Generate handles the common generation logic for all editor tools
func (tool *EditorTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("editor.EditorTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-tool.pctx.Done():
				errnie.Debug("editor.EditorTool.Generate: parent context done")
				tool.cancel()
				return
			case <-tool.ctx.Done():
				errnie.Debug("editor.EditorTool.Generate: context done")
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

// EditorReadTool implements a tool for reading files
type EditorReadTool struct {
	*EditorTool
}

// NewEditorReadTool creates a new tool for reading files
func NewEditorReadTool() *EditorReadTool {
	// Create MCP tool definition based on schema from config.yml
	readTool := mcp.NewTool(
		"read",
		mcp.WithDescription("A tool which can read files in the workspace."),
		mcp.WithString(
			"file",
			mcp.Description("The file to read."),
			mcp.Required(),
		),
	)

	ert := &EditorReadTool{
		EditorTool: NewEditorTool(),
	}

	ert.ToolBuilder.mcp = &readTool
	return ert
}

func (tool *EditorReadTool) ID() string {
	return "editor_read"
}

// Generate processes the file reading operation
func (tool *EditorReadTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.EditorTool.Generate(buffer, tool.fn)
}

// fn implements the file reading operation
func (tool *EditorReadTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("editor.EditorReadTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "read")

	// Implementation for reading files
	return artifact
}

// ToMCP returns the MCP tool definitions for the EditorReadTool
func (tool *EditorReadTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// EditorWriteTool implements a tool for writing to files
type EditorWriteTool struct {
	*EditorTool
}

// NewEditorWriteTool creates a new tool for writing to files
func NewEditorWriteTool() *EditorWriteTool {
	// Create MCP tool definition based on schema from config.yml
	writeTool := mcp.NewTool(
		"write",
		mcp.WithDescription("A tool which can write to files in the workspace."),
		mcp.WithString(
			"file",
			mcp.Description("The file to write to."),
			mcp.Required(),
		),
		mcp.WithString(
			"content",
			mcp.Description("The content to write to the file."),
			mcp.Required(),
		),
	)

	ewt := &EditorWriteTool{
		EditorTool: NewEditorTool(),
	}

	ewt.ToolBuilder.mcp = &writeTool
	return ewt
}

func (tool *EditorWriteTool) ID() string {
	return "editor_write"
}

// Generate processes the file writing operation
func (tool *EditorWriteTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.EditorTool.Generate(buffer, tool.fn)
}

// fn implements the file writing operation
func (tool *EditorWriteTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("editor.EditorWriteTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "write")

	// Implementation for writing to files
	return artifact
}

// ToMCP returns the MCP tool definitions for the EditorWriteTool
func (tool *EditorWriteTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// EditorDeleteTool implements a tool for deleting files
type EditorDeleteTool struct {
	*EditorTool
}

// NewEditorDeleteTool creates a new tool for deleting files
func NewEditorDeleteTool() *EditorDeleteTool {
	// Create MCP tool definition based on schema from config.yml
	deleteTool := mcp.NewTool(
		"delete",
		mcp.WithDescription("A tool which can delete files in the workspace."),
		mcp.WithString(
			"file",
			mcp.Description("The file to delete."),
			mcp.Required(),
		),
	)

	edt := &EditorDeleteTool{
		EditorTool: NewEditorTool(),
	}

	edt.ToolBuilder.mcp = &deleteTool
	return edt
}

func (tool *EditorDeleteTool) ID() string {
	return "editor_delete"
}

// Generate processes the file deletion operation
func (tool *EditorDeleteTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.EditorTool.Generate(buffer, tool.fn)
}

// fn implements the file deletion operation
func (tool *EditorDeleteTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("editor.EditorDeleteTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "delete")

	// Implementation for deleting files
	return artifact
}

// ToMCP returns the MCP tool definitions for the EditorDeleteTool
func (tool *EditorDeleteTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// EditorReplaceLinesTool implements a tool for replacing lines in files
type EditorReplaceLinesTool struct {
	*EditorTool
}

// NewEditorReplaceLinesTool creates a new tool for replacing lines in files
func NewEditorReplaceLinesTool() *EditorReplaceLinesTool {
	// Create MCP tool definition based on schema from config.yml
	replaceLinesTool := mcp.NewTool(
		"replace_lines",
		mcp.WithDescription("A tool which can replace lines in a file."),
		mcp.WithString(
			"file",
			mcp.Description("The file to replace lines in."),
			mcp.Required(),
		),
		mcp.WithNumber(
			"start_line",
			mcp.Description("The line number for single-line operations like insertion (1-based indexing)."),
			mcp.Required(),
		),
		mcp.WithNumber(
			"end_line",
			mcp.Description("The line number for single-line operations like insertion (1-based indexing, inclusive)."),
			mcp.Required(),
		),
		mcp.WithString(
			"content",
			mcp.Description("The content to replace the lines with."),
			mcp.Required(),
		),
	)

	erlt := &EditorReplaceLinesTool{
		EditorTool: NewEditorTool(),
	}

	erlt.ToolBuilder.mcp = &replaceLinesTool
	return erlt
}

func (tool *EditorReplaceLinesTool) ID() string {
	return "editor_replace_lines"
}

// Generate processes the line replacement operation
func (tool *EditorReplaceLinesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.EditorTool.Generate(buffer, tool.fn)
}

// fn implements the line replacement operation
func (tool *EditorReplaceLinesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("editor.EditorReplaceLinesTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "replace_lines")

	// Implementation for replacing lines
	return artifact
}

// ToMCP returns the MCP tool definitions for the EditorReplaceLinesTool
func (tool *EditorReplaceLinesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// EditorInsertLinesTool implements a tool for inserting lines into files
type EditorInsertLinesTool struct {
	*EditorTool
}

// NewEditorInsertLinesTool creates a new tool for inserting lines into files
func NewEditorInsertLinesTool() *EditorInsertLinesTool {
	// Create MCP tool definition based on schema from config.yml
	insertLinesTool := mcp.NewTool(
		"insert_lines",
		mcp.WithDescription("A tool which can insert lines into a file."),
		mcp.WithString(
			"file",
			mcp.Description("The file to insert lines into."),
			mcp.Required(),
		),
		mcp.WithString(
			"content",
			mcp.Description("The content to insert into the file."),
			mcp.Required(),
		),
	)

	eilt := &EditorInsertLinesTool{
		EditorTool: NewEditorTool(),
	}

	eilt.ToolBuilder.mcp = &insertLinesTool
	return eilt
}

func (tool *EditorInsertLinesTool) ID() string {
	return "editor_insert_lines"
}

// Generate processes the line insertion operation
func (tool *EditorInsertLinesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.EditorTool.Generate(buffer, tool.fn)
}

// fn implements the line insertion operation
func (tool *EditorInsertLinesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("editor.EditorInsertLinesTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "insert_lines")

	// Implementation for inserting lines
	return artifact
}

// ToMCP returns the MCP tool definitions for the EditorInsertLinesTool
func (tool *EditorInsertLinesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// EditorDeleteLinesTool implements a tool for deleting lines from files
type EditorDeleteLinesTool struct {
	*EditorTool
}

// NewEditorDeleteLinesTool creates a new tool for deleting lines from files
func NewEditorDeleteLinesTool() *EditorDeleteLinesTool {
	// Create MCP tool definition based on schema from config.yml
	deleteLinesTool := mcp.NewTool(
		"delete_lines",
		mcp.WithDescription("A tool which can delete lines from a file."),
		mcp.WithString(
			"file",
			mcp.Description("The file to delete lines from."),
			mcp.Required(),
		),
		mcp.WithNumber(
			"start_line",
			mcp.Description("The line number for single-line operations like insertion (1-based indexing)."),
			mcp.Required(),
		),
		mcp.WithNumber(
			"end_line",
			mcp.Description("The line number for single-line operations like insertion (1-based indexing, inclusive)."),
			mcp.Required(),
		),
	)

	edlt := &EditorDeleteLinesTool{
		EditorTool: NewEditorTool(),
	}

	edlt.ToolBuilder.mcp = &deleteLinesTool
	return edlt
}

func (tool *EditorDeleteLinesTool) ID() string {
	return "editor_delete_lines"
}

// Generate processes the line deletion operation
func (tool *EditorDeleteLinesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.EditorTool.Generate(buffer, tool.fn)
}

// fn implements the line deletion operation
func (tool *EditorDeleteLinesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("editor.EditorDeleteLinesTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "delete_lines")

	// Implementation for deleting lines
	return artifact
}

// ToMCP returns the MCP tool definitions for the EditorDeleteLinesTool
func (tool *EditorDeleteLinesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// EditorReadLinesTool implements a tool for reading lines from files
type EditorReadLinesTool struct {
	*EditorTool
}

// NewEditorReadLinesTool creates a new tool for reading lines from files
func NewEditorReadLinesTool() *EditorReadLinesTool {
	// Create MCP tool definition based on schema from config.yml
	readLinesTool := mcp.NewTool(
		"read_lines",
		mcp.WithDescription("A tool which can read lines from a file."),
		mcp.WithString(
			"file",
			mcp.Description("The file to read lines from."),
			mcp.Required(),
		),
		mcp.WithNumber(
			"start_line",
			mcp.Description("The line number for single-line operations like insertion (1-based indexing)."),
			mcp.Required(),
		),
		mcp.WithNumber(
			"end_line",
			mcp.Description("The line number for single-line operations like insertion (1-based indexing, inclusive)."),
			mcp.Required(),
		),
	)

	erlt := &EditorReadLinesTool{
		EditorTool: NewEditorTool(),
	}

	erlt.ToolBuilder.mcp = &readLinesTool
	return erlt
}

func (tool *EditorReadLinesTool) ID() string {
	return "editor_read_lines"
}

// Generate processes the line reading operation
func (tool *EditorReadLinesTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.EditorTool.Generate(buffer, tool.fn)
}

// fn implements the line reading operation
func (tool *EditorReadLinesTool) fn(artifact *datura.Artifact) *datura.Artifact {
	errnie.Debug("editor.EditorReadLinesTool.fn")

	// Set operation for processing
	artifact.SetMetaValue("operation", "read_lines")

	// Implementation for reading lines
	return artifact
}

// ToMCP returns the MCP tool definitions for the EditorReadLinesTool
func (tool *EditorReadLinesTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}
