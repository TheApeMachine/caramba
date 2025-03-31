package tool

import (
	"fmt"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

func NewCapnpTool(toolName string) (*Tool, error) {
	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		tool  Tool
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); err != nil {
		return nil, errnie.Error(err)
	}

	if tool, err = NewRootTool(seg); err != nil {
		return nil, errnie.Error(err)
	}

	base := fmt.Sprintf("tools.schemas.%s", toolName)

	// Get function details from config
	name := tweaker.Get(base+".function.name", toolName)
	description := tweaker.Get(base+".function.description", "")

	function, err := tool.NewFunction()
	if err != nil {
		return nil, errnie.Error(err)
	}

	if err := function.SetName(name); err != nil {
		return nil, errnie.Error(err)
	}

	if err := function.SetDescription(description); err != nil {
		return nil, errnie.Error(err)
	}

	parameters, err := function.NewParameters()
	if err != nil {
		return nil, errnie.Error(err)
	}

	if err := parameters.SetType("object"); err != nil {
		return nil, errnie.Error(err)
	}

	// Get properties from config
	if props := tweaker.GetStringMap(base + ".properties"); len(props) > 0 {
		properties, err := parameters.NewProperties(int32(len(props)))
		if err != nil {
			return nil, errnie.Error(err)
		}

		i := 0
		for propName := range props {
			propBase := fmt.Sprintf("%s.properties.%s", base, propName)

			property, err := NewProperty(parameters.Segment())
			if err != nil {
				return nil, errnie.Error(err)
			}

			if err := property.SetName(propName); err != nil {
				return nil, errnie.Error(err)
			}

			if err := property.SetType(tweaker.Get(propBase+".type", "string")); err != nil {
				return nil, errnie.Error(err)
			}

			if err := property.SetDescription(tweaker.Get(propBase+".description", "")); err != nil {
				return nil, errnie.Error(err)
			}

			// Handle enum/options if present
			if options := tweaker.GetStringSlice(propBase + ".options"); len(options) > 0 {
				enum, err := property.NewEnum(int32(len(options)))
				if err != nil {
					return nil, errnie.Error(err)
				}
				for j, opt := range options {
					if err := enum.Set(j, opt); err != nil {
						return nil, errnie.Error(err)
					}
				}
			}

			if err := properties.Set(i, property); err != nil {
				return nil, errnie.Error(err)
			}
			i++
		}
	}

	if required := tweaker.GetStringSlice(base + ".required"); len(required) > 0 {
		requiredList, err := parameters.NewRequired(int32(len(required)))
		if err != nil {
			return nil, errnie.Error(err)
		}
		for i, req := range required {
			if err := requiredList.Set(i, req); err != nil {
				return nil, errnie.Error(err)
			}
		}
	}

	return &tool, nil
}
