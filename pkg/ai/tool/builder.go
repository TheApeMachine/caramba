package tool

import (
	"bytes"
	"errors"
	"io"

	"github.com/google/uuid"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools"
)

type ToolOption func(*Tool) error

// New creates a new tool with the provided options
func New(options ...ToolOption) *Tool {
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

	if errnie.Error(tool.SetUuid(uuid.New().String())) != nil {
		return nil
	}

	tool.ToState(errnie.StateReady)

	// Apply all options
	for _, option := range options {
		if err := option(&tool); errnie.Error(err) != nil {
			return nil
		}
	}

	return &tool
}

func WithBytes(b []byte) ToolOption {
	errnie.Trace("tool.WithBytes")

	return func(tool *Tool) error {
		if _, err := io.Copy(tool, bytes.NewBuffer(b)); errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		return nil
	}
}

func WithMCPTool(mcpTools ...tools.ToolType) ToolOption {
	errnie.Trace("tool.WithMCPTool")

	return func(tool *Tool) (err error) {
		var (
			ops      Operation_List
			opIdx    int32 = 0
			totalOps int32 = 0
		)

		// Get existing operations if they exist
		if ops, err = tool.Operations(); err == nil {
			totalOps = int32(ops.Len())
		} else {
			// Initialize if list doesn't exist
			if ops, err = NewOperation_List(tool.Segment(), int32(len(mcpTools))); err != nil {
				return errnie.Error(err)
			}
		}

		if totalOps < int32(ops.Len()+len(mcpTools)) {
			requiredSize := totalOps + int32(len(mcpTools))
			if ops, err = NewOperation_List(tool.Segment(), requiredSize); err != nil {
				return errnie.Error(err)
			}
			totalOps = 0
		}

		opIdx = totalOps

		for _, mcpTool := range mcpTools {
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
			params, err := NewParameter_List(tool.Segment(), int32(len(paramsMap)))
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
						enumList, err := capnp.NewTextList(tool.Segment(), int32(len(pEnum)))
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
			required, err := capnp.NewTextList(tool.Segment(), int32(len(requiredSlice)))
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
		if err = tool.SetOperations(ops); err != nil {
			return errnie.Error(err)
		}

		return nil
	}
}
