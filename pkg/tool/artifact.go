package tool

import (
	"fmt"
	"reflect"

	"capnproto.org/go/capnp/v3"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func root() (*Artifact, error) {
	arena := capnp.SingleSegment(nil)

	_, seg, err := capnp.NewMessage(arena)
	if errnie.Error(err) != nil {
		return nil, err
	}

	artfct, err := NewRootArtifact(seg)
	if errnie.Error(err) != nil {
		return nil, err
	}

	return &artfct, nil
}

/*
New creates a new artifact with the given origin, role, scope, and data.
*/
func New() *Artifact {
	var (
		err      error
		artifact *Artifact
	)

	if artifact, err = root(); errnie.Error(err) != nil {
		return nil
	}

	artifact.SetType("function")
	return artifact
}

func (artifact *Artifact) WithFunction(
	name, description string,
	parameters map[string]any,
) *Artifact {
	function, err := NewFunction(artifact.Segment())
	if errnie.Error(err) != nil {
		return nil
	}

	paramList, err := function.NewParameters(int32(len(parameters)))
	if errnie.Error(err) != nil {
		return nil
	}

	i := 0
	for key, value := range parameters {
		param, err := NewParameter(artifact.Segment())
		if errnie.Error(err) != nil {
			return nil
		}

		if err := param.SetType(reflect.TypeOf(value).String()); errnie.Error(err) != nil {
			return nil
		}

		propList, err := param.NewProperties(1)
		if errnie.Error(err) != nil {
			return nil
		}

		prop, err := NewProperty(artifact.Segment())
		if errnie.Error(err) != nil {
			return nil
		}

		if err := prop.SetName(key); errnie.Error(err) != nil {
			return nil
		}

		strValue := fmt.Sprintf("%v", value)
		if err := prop.SetDescription(strValue); errnie.Error(err) != nil {
			return nil
		}

		propList.Set(0, prop)

		required, err := param.NewRequired(1)
		if errnie.Error(err) != nil {
			return nil
		}
		required.Set(0, key)

		paramList.Set(i, param)
		i++
	}

	function.SetName(name)
	function.SetDescription(description)
	function.SetParameters(paramList)

	artifact.SetFunction(function)
	return artifact
}

func (artifact *Artifact) ToMCP() *mcp.Tool {
	var (
		err      error
		function Function
		name     string
		desc     string
	)

	if function, err = artifact.Function(); errnie.Error(err) != nil {
		return nil
	}

	if name, err = function.Name(); errnie.Error(err) != nil {
		return nil
	}

	if desc, err = function.Description(); errnie.Error(err) != nil {
		return nil
	}

	tool := mcp.NewTool(
		name,
		mcp.WithDescription(desc),
	)

	return &tool
}
