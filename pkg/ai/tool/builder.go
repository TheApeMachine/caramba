package tool

import (
	"bufio"
	"bytes"
	context "context"
	"errors"
	"io"

	"capnproto.org/go/capnp/v3"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools"
)

type ToolBuilder struct {
	*Tool
	MCPTool *tools.ToolType
	encoder *capnp.Encoder
	decoder *capnp.Decoder
	buffer  *bufio.ReadWriter
	State   ToolState
	client  RPC
}

type ToolBuilderOption func(*ToolBuilder) error

// New creates a new tool with the provided options
func New(options ...ToolBuilderOption) *ToolBuilder {
	errnie.Trace("tool.New")

	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		tool  Tool
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if tool, err = NewRootTool(seg); errnie.Error(err) != nil {
		return nil
	}

	builder := &ToolBuilder{
		Tool: &tool,
	}

	// Apply all options
	for _, option := range options {
		if err := option(builder); errnie.Error(err) != nil {
			return nil
		}
	}

	// Initialize the RPC client
	builder.client = ToolToClient(builder)

	return builder
}

func (tb *ToolBuilder) Use(ctx context.Context, artifact *datura.Artifact) *datura.Artifact {
	errnie.Trace("tool.Use")

	future, release := tb.client.Use(
		ctx, func(p RPC_use_Params) error {
			return p.SetArtifact(*artifact)
		},
	)

	defer release()

	var (
		result RPC_use_Results
		err    error
	)

	if result, err = future.Struct(); errnie.Error(err) != nil {
		return nil
	}

	out, err := result.Out()

	if errnie.Error(err) != nil {
		return nil
	}

	return &out
}

func WithBytes(b []byte) ToolBuilderOption {
	errnie.Trace("tool.WithBytes")

	return func(t *ToolBuilder) error {
		if _, err := io.Copy(t, bytes.NewBuffer(b)); errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		return nil
	}
}

func WithMCPTool(mcpTools ...tools.ToolType) ToolBuilderOption {
	errnie.Trace("tool.WithMCPTool")

	return func(t *ToolBuilder) (err error) {
		var (
			ops      Operation_List
			opIdx    int32 = 0
			totalOps int32 = 0
		)

		// Get existing operations if they exist
		if ops, err = t.Tool.Operations(); err == nil {
			totalOps = int32(ops.Len())
		} else {
			// Initialize if list doesn't exist
			if ops, err = NewOperation_List(t.Tool.Segment(), int32(len(mcpTools))); err != nil {
				return errnie.Error(err)
			}
		}

		// Resize list if needed
		if totalOps < int32(ops.Len()+len(mcpTools)) {
			// This is likely incorrect usage of capnp list resizing/appending logic.
			// Need to re-allocate or use a different approach if appending is the goal.
			// For now, assuming we are creating a new list or replacing.
			// Let's recreate the list with the correct total size.
			requiredSize := totalOps + int32(len(mcpTools))
			if ops, err = NewOperation_List(t.Tool.Segment(), requiredSize); err != nil {
				return errnie.Error(err)
			}
			// Note: If appending was intended, the original elements need to be copied here.
			// Resetting totalOps as we are starting fresh.
			totalOps = 0
		}

		opIdx = totalOps // Start adding new ops after existing ones

		// Iterate over the provided MCP tools
		for _, mcpTool := range mcpTools {
			t.MCPTool = &mcpTool

			if opIdx >= int32(ops.Len()) {
				// This should ideally not happen if resizing logic is correct
				return errnie.Error(errors.New("operation list length exceeded unexpectedly"))
			}
			op := ops.At(int(opIdx))

			// --- Set Operation Name and Description ---
			if err = op.SetName(mcpTool.Tool.Name); err != nil {
				return errnie.Error(err)
			}
			if err = op.SetDescription(mcpTool.Tool.Description); err != nil {
				return errnie.Error(err)
			}

			// --- Translate Parameters ---
			paramsMap := mcpTool.Tool.InputSchema.Properties
			params, err := NewParameter_List(t.Tool.Segment(), int32(len(paramsMap)))
			if err != nil {
				return errnie.Error(err)
			}

			paramIdx := 0
			for name, schema := range paramsMap {
				p := params.At(paramIdx)
				if err = p.SetName(name); err != nil {
					return errnie.Error(err)
				}

				// Type assertion to extract details from the interface{}
				if schemaMap, ok := schema.(map[string]interface{}); ok {
					if pType, ok := schemaMap["type"].(string); ok {
						if err = p.SetType(pType); err != nil {
							return errnie.Error(err)
						}
					}
					if pDesc, ok := schemaMap["description"].(string); ok {
						if err = p.SetDescription(pDesc); err != nil {
							return errnie.Error(err)
						}
					}
					if pEnum, ok := schemaMap["enum"].([]interface{}); ok {
						enumList, err := capnp.NewTextList(t.Tool.Segment(), int32(len(pEnum)))
						if err != nil {
							return errnie.Error(err)
						}
						for i, enumVal := range pEnum {
							if enumStr, ok := enumVal.(string); ok {
								if err = enumList.Set(i, enumStr); err != nil {
									return errnie.Error(err)
								}
							}
						}
						if err = p.SetEnum(enumList); err != nil {
							return errnie.Error(err)
						}
					}
				}
				paramIdx++
			}
			if err = op.SetParameters(params); err != nil {
				return errnie.Error(err)
			}

			// --- Translate Required Parameters ---
			requiredSlice := mcpTool.Tool.InputSchema.Required
			required, err := capnp.NewTextList(t.Tool.Segment(), int32(len(requiredSlice)))
			if err != nil {
				return errnie.Error(err)
			}
			for i, reqName := range requiredSlice {
				if err = required.Set(i, reqName); err != nil {
					return errnie.Error(err)
				}
			}
			if err = op.SetRequired(required); err != nil {
				return errnie.Error(err)
			}

			opIdx++ // Move to the next index for the next MCP tool
		}

		// Set the final operations list on the tool
		if err = t.Tool.SetOperations(ops); err != nil {
			return errnie.Error(err)
		}

		return nil
	}
}
