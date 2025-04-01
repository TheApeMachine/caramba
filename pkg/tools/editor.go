package tools

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools/editor"
)

func init() {
	fmt.Println("tools.editor.init")
	provider.RegisterTool("editor")
}

// EditorBaseTool provides common functionality for all editor tools
type EditorBaseTool struct {
	pctx      context.Context
	ctx       context.Context
	cancel    context.CancelFunc
	client    *editor.Client
	Schema    *provider.Tool
	operation string
}

// Generate handles the common generation logic for all editor tools
func (ebt *EditorBaseTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	errnie.Debug("editor.EditorBaseTool.Generate." + ebt.operation)

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-ebt.pctx.Done():
			errnie.Debug("editor.EditorBaseTool.Generate.pctx.Done." + ebt.operation)
			ebt.cancel()
			return
		case <-ebt.ctx.Done():
			errnie.Debug("editor.EditorBaseTool.Generate.ctx.Done." + ebt.operation)
			ebt.cancel()
			return
		case artifact := <-buffer:
			artifact.SetMetaValue("operation", ebt.operation)
			out <- artifact
		}
	}()

	return out
}

// NewEditorBaseTool creates a new base editor tool with the specified schema and operation
func NewEditorBaseTool(schemaName, operation string) *EditorBaseTool {
	client := editor.NewClient()

	return &EditorBaseTool{
		client:    client,
		Schema:    GetToolSchema(schemaName),
		operation: operation,
	}
}

// WithCancelBase sets the parent context for an editor base tool
func WithCancelBase(ctx context.Context) func(*EditorBaseTool) {
	return func(tool *EditorBaseTool) {
		tool.pctx = ctx
	}
}

// EditorReadTool implements a tool for reading files
type EditorReadTool struct {
	*EditorBaseTool
}

// NewEditorReadTool creates a new tool for reading files
func NewEditorReadTool() *EditorReadTool {
	return &EditorReadTool{
		EditorBaseTool: NewEditorBaseTool("editor.read", "read"),
	}
}

// Generate forwards the generation to the base tool
func (ert *EditorReadTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	return ert.EditorBaseTool.Generate(buffer)
}

// WithCancelEditorReadTool sets the parent context for an editor read tool
func WithCancelEditorReadTool(ctx context.Context) func(*EditorReadTool) {
	return func(tool *EditorReadTool) {
		WithCancelBase(ctx)(tool.EditorBaseTool)
	}
}

// EditorWriteTool implements a tool for writing to files
type EditorWriteTool struct {
	*EditorBaseTool
}

// NewEditorWriteTool creates a new tool for writing to files
func NewEditorWriteTool() *EditorWriteTool {
	return &EditorWriteTool{
		EditorBaseTool: NewEditorBaseTool("editor.write", "write"),
	}
}

// Generate forwards the generation to the base tool
func (ewt *EditorWriteTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	return ewt.EditorBaseTool.Generate(buffer)
}

// WithCancelEditorWriteTool sets the parent context for an editor write tool
func WithCancelEditorWriteTool(ctx context.Context) func(*EditorWriteTool) {
	return func(tool *EditorWriteTool) {
		WithCancelBase(ctx)(tool.EditorBaseTool)
	}
}

// EditorDeleteTool implements a tool for deleting files
type EditorDeleteTool struct {
	*EditorBaseTool
}

// NewEditorDeleteTool creates a new tool for deleting files
func NewEditorDeleteTool() *EditorDeleteTool {
	return &EditorDeleteTool{
		EditorBaseTool: NewEditorBaseTool("editor.delete", "delete"),
	}
}

// Generate forwards the generation to the base tool
func (edt *EditorDeleteTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	return edt.EditorBaseTool.Generate(buffer)
}

// WithCancelEditorDeleteTool sets the parent context for an editor delete tool
func WithCancelEditorDeleteTool(ctx context.Context) func(*EditorDeleteTool) {
	return func(tool *EditorDeleteTool) {
		WithCancelBase(ctx)(tool.EditorBaseTool)
	}
}

// EditorReplaceLinesTool implements a tool for replacing lines in files
type EditorReplaceLinesTool struct {
	*EditorBaseTool
}

// NewEditorReplaceLinesTool creates a new tool for replacing lines in files
func NewEditorReplaceLinesTool() *EditorReplaceLinesTool {
	return &EditorReplaceLinesTool{
		EditorBaseTool: NewEditorBaseTool("editor.replace_lines", "replace_lines"),
	}
}

// Generate forwards the generation to the base tool
func (erlt *EditorReplaceLinesTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	return erlt.EditorBaseTool.Generate(buffer)
}

// WithCancelEditorReplaceLinesTool sets the parent context for an editor replace lines tool
func WithCancelEditorReplaceLinesTool(ctx context.Context) func(*EditorReplaceLinesTool) {
	return func(tool *EditorReplaceLinesTool) {
		WithCancelBase(ctx)(tool.EditorBaseTool)
	}
}

// EditorInsertLinesTool implements a tool for inserting lines into files
type EditorInsertLinesTool struct {
	*EditorBaseTool
}

// NewEditorInsertLinesTool creates a new tool for inserting lines into files
func NewEditorInsertLinesTool() *EditorInsertLinesTool {
	return &EditorInsertLinesTool{
		EditorBaseTool: NewEditorBaseTool("editor.insert_lines", "insert_lines"),
	}
}

// Generate forwards the generation to the base tool
func (eilt *EditorInsertLinesTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	return eilt.EditorBaseTool.Generate(buffer)
}

// WithCancelEditorInsertLinesTool sets the parent context for an editor insert lines tool
func WithCancelEditorInsertLinesTool(ctx context.Context) func(*EditorInsertLinesTool) {
	return func(tool *EditorInsertLinesTool) {
		WithCancelBase(ctx)(tool.EditorBaseTool)
	}
}

// EditorDeleteLinesTool implements a tool for deleting lines from files
type EditorDeleteLinesTool struct {
	*EditorBaseTool
}

// NewEditorDeleteLinesTool creates a new tool for deleting lines from files
func NewEditorDeleteLinesTool() *EditorDeleteLinesTool {
	return &EditorDeleteLinesTool{
		EditorBaseTool: NewEditorBaseTool("editor.delete_lines", "delete_lines"),
	}
}

// Generate forwards the generation to the base tool
func (edlt *EditorDeleteLinesTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	return edlt.EditorBaseTool.Generate(buffer)
}

// WithCancelEditorDeleteLinesTool sets the parent context for an editor delete lines tool
func WithCancelEditorDeleteLinesTool(ctx context.Context) func(*EditorDeleteLinesTool) {
	return func(tool *EditorDeleteLinesTool) {
		WithCancelBase(ctx)(tool.EditorBaseTool)
	}
}

// EditorReadLinesTool implements a tool for reading lines from files
type EditorReadLinesTool struct {
	*EditorBaseTool
}

// NewEditorReadLinesTool creates a new tool for reading lines from files
func NewEditorReadLinesTool() *EditorReadLinesTool {
	return &EditorReadLinesTool{
		EditorBaseTool: NewEditorBaseTool("editor.read_lines", "read_lines"),
	}
}

// Generate forwards the generation to the base tool
func (erlt *EditorReadLinesTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	return erlt.EditorBaseTool.Generate(buffer)
}

// WithCancelEditorReadLinesTool sets the parent context for an editor read lines tool
func WithCancelEditorReadLinesTool(ctx context.Context) func(*EditorReadLinesTool) {
	return func(tool *EditorReadLinesTool) {
		WithCancelBase(ctx)(tool.EditorBaseTool)
	}
}
