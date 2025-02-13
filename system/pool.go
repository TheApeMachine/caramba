package system

import (
	"regexp"
	"strings"
	"sync"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/agent"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/tweaker"
	"github.com/theapemachine/caramba/types"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

var poolOnce sync.Once
var poolInstance *Pool

type Entity struct {
	Config    types.Config
	Generator types.Generator
	Toolset   *tools.Toolset
}

/*
Pool is an ambient context which provides a reference to every agent
that is currently active in the system
*/
type Pool struct {
	entities map[string]*Entity
	selected *Entity
	filo     []*Entity
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

	errnie.Info("generating agent: " + pool.selected.Config.Name())

	go func() {
		defer close(pool.out)

		accumulator := stream.NewAccumulator()
		accumulator.After(pool.after)
		pool.selected.Generator.SetStatus(types.AgentStatusBusy)

		for pool.selected.Generator.Status() == types.AgentStatusBusy {
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
	if pool.selected == nil {
		errnie.Warn("no agent selected in after function")
		return
	}

	errnie.Info("post-generation processing")
	valid, err := pool.validate(response)
	if !valid {
		pool.selected.Config.Thread().AddMessage(
			provider.NewMessage(provider.RoleAssistant, tweaker.GetIteration(
				pool.selected.Config.Name(),
				pool.selected.Config.Role(),
				pool.selected.Generator.Ctx().Iteration(),
				utils.JoinWith("\n",
					pool.selected.Generator.Accumulator().String(),
					err,
				),
			)),
		)

		pool.logContext()
		return
	}

	pool.toolcalls(pool.selected.Generator.Accumulator())

	pool.selected.Config.Thread().AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			tweaker.GetIteration(
				pool.selected.Config.Name(),
				pool.selected.Config.Role(),
				pool.selected.Generator.Ctx().Iteration(),
				pool.selected.Generator.Accumulator().String(),
			),
		),
	)

	pool.logContext()
}

/*
logContextm ...
*/
func (pool *Pool) logContext() {
	out := []string{
		"<<START AGENT: " + pool.selected.Config.Name() + ">>",
	}
	for _, message := range pool.selected.Config.Thread().Messages {
		out = append(out, message.Content)
	}
	out = append(out, "<<END AGENT: "+pool.selected.Config.Name()+">>\n")

	errnie.Log(strings.Join(out, "\n"))
}

/*
toolcalls processes any tool invocations found in the generated response.
It extracts JSON blocks from the response and executes the corresponding
tool calls.
*/
func (pool *Pool) toolcalls(accumulator *stream.Accumulator) *stream.Accumulator {
	errnie.Info("processing toolcalls")

	blocks := utils.ExtractJSONBlocks(pool.selected.Generator.Accumulator().String())
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
		agents := make([]types.Generator, 0)

		for _, agent := range pool.selected.Generator.Agents() {
			agents = append(agents, agent)
		}

		return pool.selected.Toolset.Use(accumulator, toolname, args, agents...)
	case "send":
		if _, ok := args["to"].(string); !ok {
			errnie.Warn("invalid receiver")
			return accumulator
		}

		var (
			to string
			ok bool
		)

		if to, ok = args["to"].(string); !ok {
			errnie.Warn("invalid receiver", "to", args["to"])
			return accumulator
		}

		// Store current agent in FILO stack
		pool.filo = append(pool.filo, pool.selected)

		// Switch to the target agent
		if targetEntity, exists := pool.entities[to]; exists {
			pool.selected = targetEntity
		} else if agent, exists := pool.selected.Generator.Agents()[to]; exists {
			// If the agent exists in the current agent's hierarchy, create a new entity
			pool.entities[to] = &Entity{
				Config:    agent.Ctx().Config(),
				Generator: agent,
				Toolset:   tools.NewToolset(&tools.Agent{}),
			}
			pool.selected = pool.entities[to]
		}

		out := pool.selected.Toolset.Use(
			accumulator,
			toolname,
			args,
			pool.selected.Generator,
		)

		// Restore previous agent
		pool.selected, pool.filo = pool.filo[len(pool.filo)-1], pool.filo[:len(pool.filo)-1]
		return out
	case "team":
		out := pool.selected.Toolset.Use(accumulator, toolname, args, pool.selected.Generator)
		agents := pool.selected.Generator.Agents()

		// Store current agent in FILO stack
		pool.filo = append(pool.filo, pool.selected)

		for name, generator := range agents {
			pool.entities[name] = &Entity{
				Config:    generator.Ctx().Config(),
				Generator: generator,
				Toolset:   tools.NewToolset(&tools.Agent{}),
			}

			// Switch to the new agent temporarily
			pool.selected = pool.entities[name]

			// Process the teamlead's response through their thread
			teamResponse := ""
			for event := range pool.selected.Generator.Generate(provider.NewMessage(
				provider.RoleUser,
				viper.GetViper().GetString("prompts.templates.teambuilding"),
			)) {
				if event.Type == provider.EventChunk {
					teamResponse += event.Text
				}
			}

			// Add the teamlead's response to their thread
			pool.selected.Config.Thread().AddMessage(provider.NewMessage(
				provider.RoleAssistant,
				teamResponse,
			))

			// Add a ready message to the coordinator's accumulator
			accumulator.Append(utils.QuickWrap("READY", utils.JoinWith("\n",
				"AGENT : "+name,
				"ROLE  : teamlead",
				"STATUS: ready for instructions",
			), 1))
		}

		// Restore previous agent
		pool.selected, pool.filo = pool.filo[len(pool.filo)-1], pool.filo[:len(pool.filo)-1]

		return out
	case "agent":
		out := pool.selected.Toolset.Use(accumulator, toolname, args, pool.selected.Generator)
		agents := pool.selected.Generator.Agents()

		for name, generator := range agents {
			pool.entities[name] = &Entity{
				Config:    generator.Ctx().Config(),
				Generator: generator,
				Toolset:   tools.NewToolset(&tools.Agent{}),
			}
		}

		return out
	default:
		return pool.selected.Toolset.Use(accumulator, toolname, args, pool.selected.Generator)
	}
}

func (pool *Pool) validate(response string) (bool, string) {
	errnie.Debug("validating response")

	// Extract content between <details> tags
	detailsRe := regexp.MustCompile(`<details>([\s\S]*?)</details>`)
	matches := detailsRe.FindAllString(response, -1)

	// Check each <details> block
	summaryRe := regexp.MustCompile(`<summary>([\s\S]*?)</summary>`)
	for _, match := range matches {
		summaryMatches := summaryRe.FindStringSubmatch(match)
		if len(summaryMatches) > 1 {
			// Get the text between <summary> tags
			summaryText := strings.TrimSpace(summaryMatches[1])
			if summaryText != "scratchpad" {
				return false, "[ERROR] do not add your own details blocks, besides 'scratchpad' [/ERROR]"
			}
		}
	}

	return true, ""
}
