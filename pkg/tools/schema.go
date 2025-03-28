package tools

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

// GetToolSchema returns a tool schema from the configuration.
func GetToolSchema(toolName string) *provider.Tool {
	base := fmt.Sprintf("tools.schemas.%s", toolName)

	// Get function details
	name := tweaker.Get(base+".function.name", toolName)
	description := tweaker.Get(base+".function.description", "")

	// Create tool with function
	opts := []provider.OptionTool{provider.WithFunction(name, description)}

	// Get properties
	if props := tweaker.GetStringMap(base + ".properties"); len(props) > 0 {
		for propName := range props {
			propBase := fmt.Sprintf("%s.properties.%s", base, propName)
			propType := tweaker.Get(propBase+".type", "string")
			propDesc := tweaker.Get(propBase+".description", "")
			var options []any
			if opts := tweaker.GetStringSlice(propBase + ".options"); len(opts) > 0 {
				for _, opt := range opts {
					options = append(options, opt)
				}
			}
			opts = append(opts, provider.WithProperty(propName, propType, propDesc, options))
		}
	}

	return provider.NewTool(opts...)
}
