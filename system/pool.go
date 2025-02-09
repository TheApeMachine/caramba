package system

import (
	"strings"
	"sync"

	"github.com/theapemachine/caramba/agent"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/tweaker"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

var poolOnce sync.Once
var poolInstance *Pool

type Entity struct {
	Config    *agent.Config
	Generator *agent.Generator
	Toolset   *tools.Toolset
}

/*
Pool is an ambient context which provides a reference to every agent
that is currently active in the system
*/
type Pool struct {
	entities map[string]*Entity
	selected *Entity
	out      chan *provider.Event
}

func NewPool() *Pool {
	poolOnce.Do(func() {
		poolInstance = &Pool{
			entities: make(map[string]*Entity),
			out:      make(chan *provider.Event),
		}
	})
	return poolInstance
}

func (pool *Pool) Add(systemPrompt, role, name string, toolset *tools.Toolset) {
	config := agent.NewConfig(
		systemPrompt,
		role,
		name,
		toolset.String(),
	)

	generator := agent.NewGenerator(
		config,
		provider.NewBalancedProvider(),
	)

	pool.entities[name] = &Entity{
		Config:    config,
		Generator: generator,
		Toolset:   toolset,
	}
}

func (pool *Pool) Select(name string) *Pool {
	pool.selected = pool.entities[name]
	return pool
}

func (pool *Pool) Generate(message *provider.Message) <-chan *provider.Event {
	if pool.selected == nil {
		errnie.Warn("no agent selected")
		return nil
	}

	errnie.Info("generating agent: " + pool.selected.Config.Name)

	go func() {
		defer close(pool.out)

		accumulator := stream.NewAccumulator()
		accumulator.After(pool.after)

		for {
			for event := range accumulator.Generate(
				pool.selected.Generator.Generate(message),
			) {
				pool.out <- event
			}
		}
	}()

	return pool.out
}

/*
after performs post-generation processing steps. It handles tool calls and
updates the conversation thread with the generated response.
*/
func (pool *Pool) after(response string) {
	errnie.Info("post-generation processing")

	accumulator := pool.toolcalls(stream.NewAccumulator())

	pool.selected.Config.Thread.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			tweaker.GetIteration(
				pool.selected.Config.Name,
				pool.selected.Config.Role,
				pool.selected.Generator.Ctx.Iteration,
				utils.JoinWith("\n\n",
					pool.selected.Generator.Accumulator.String(),
					accumulator.String(),
				),
			),
		),
	)

	out := []string{
		"<<START AGENT: " + pool.selected.Config.Name + ">>",
	}
	for _, message := range pool.selected.Config.Thread.Messages {
		out = append(out, message.Content)
	}
	out = append(out, "<<END AGENT: "+pool.selected.Config.Name+">>\n")

	errnie.Log(strings.Join(out, "\n"))
}

/*
toolcalls processes any tool invocations found in the generated response.
It extracts JSON blocks from the response and executes the corresponding
tool calls.
*/
func (pool *Pool) toolcalls(accumulator *stream.Accumulator) *stream.Accumulator {
	errnie.Info("processing toolcalls")

	blocks := utils.ExtractJSONBlocks(pool.selected.Generator.Accumulator.String())
	for _, block := range blocks {
		if toolname, ok := block["tool"].(string); ok {
			if args, ok := block["args"].(map[string]any); ok {
				pool.toolcall(toolname, args, accumulator)
			}
		}
	}

	return accumulator
}

/*
toolcall executes a specific tool with the provided arguments and updates
the generator's status based on the tool's response.

Parameters:

	toolname: The name of the tool to execute
	args: A map of arguments to pass to the tool

Returns:

	string: The result of the tool execution
*/
func (pool *Pool) toolcall(
	toolname string, args map[string]any, accumulator *stream.Accumulator,
) *stream.Accumulator {
	errnie.Info("processing toolcall: " + toolname)

	switch toolname {
	case "break":
		out := pool.selected.Toolset.Use(accumulator, toolname, args, pool.selected.Generator)
		pool.selected = nil
		return out
	case "show":
		agents := make([]*agent.Generator, 0)

		for _, agent := range pool.selected.Generator.Agents {
			agents = append(agents, agent)
		}

		return pool.selected.Toolset.Use(accumulator, toolname, args, agents...)
	default:
		return pool.selected.Toolset.Use(accumulator, toolname, args, pool.selected.Generator)
	}
}
