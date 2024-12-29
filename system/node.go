package system

import (
	"context"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
)

/*
Node represents an Agent as part of a Graph.
*/
type Node struct {
	ID       string    `json:"id" jsonschema:"title=ID,description=Unique identifier for the node,required"`
	Agent    *ai.Agent `json:"agent" jsonschema:"title=Agent,description=The agent that this node represents,required"`
	Parallel bool      `json:"parallel" jsonschema:"title=Parallel,description=Whether this node should be run in parallel,required"`
}

func NewNode(id string, agent *ai.Agent, parallel bool) *Node {
	return &Node{
		ID:       id,
		Agent:    agent,
		Parallel: parallel,
	}
}

func (node *Node) Generate(ctx context.Context, message *provider.Message) <-chan provider.Event {
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		for event := range node.Agent.Generate(ctx, message) {
			out <- event
		}
	}()

	return out
}
