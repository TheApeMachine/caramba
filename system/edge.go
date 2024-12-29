package system

import "github.com/theapemachine/caramba/utils"

type DirectionType string

const (
	DirectionTypeIn   DirectionType = "in"
	DirectionTypeOut  DirectionType = "out"
	DirectionTypeBoth DirectionType = "both"
)

/*
Edge represents a connection between two Nodes, which in
turn can be seen as a communication channel between two Agents.
*/
type Edge struct {
	From      string        `json:"from" jsonschema:"title=From,description=The source Node,required"`
	To        string        `json:"to" jsonschema:"title=To,description=The destination Node,required"`
	Direction DirectionType `json:"direction" jsonschema:"title=Direction,description=The direction of the Edge,enum=in,enum=out,enum=both,required"`
}

func (edge *Edge) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Edge]()
}
