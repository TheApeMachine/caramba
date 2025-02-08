package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tweaker"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
Generator is an abstraction which provides the language generation capabilities
to an Agent.
*/
type Generator struct {
	config      *Config
	provider    provider.Provider
	ctx         *Context
	accumulator *stream.Accumulator
	status      AgentStatus
}

/*
NewGenrator takes an agent Config, and a Provider, which are the main ingredients
needed by all LLM providers, so they can generate responses.
*/
func NewGenerator(
	config *Config, prvdr provider.Provider,
) *Generator {
	return &Generator{
		config:      config,
		provider:    prvdr,
		ctx:         NewContext(config),
		accumulator: stream.NewAccumulator(),
		status:      AgentStatusIdle,
	}
}

func (generator *Generator) Generate(
	message *provider.Message,
) <-chan *provider.Event {
	out := make(chan *provider.Event)

	go func() {
		defer close(out)

		generator.status = AgentStatusBusy
		generator.ctx.AddUserPrompt(message)

		generator.accumulator.After(func(str string) {
			generator.after()
		})

		for {
			generator.ctx.iteration++

			for event := range generator.accumulator.Generate(
				generator.provider.Generate(generator.ctx.params),
			) {
				out <- event
			}

			for _, msg := range generator.ctx.config.Thread.Messages {
				errnie.Log(msg.Content)
			}
		}
	}()

	return out
}

func (generator *Generator) after() {
	generator.toolcalls()

	generator.ctx.config.Thread.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			tweaker.GetIteration(
				generator.ctx.config.Name,
				generator.ctx.config.Role,
				generator.ctx.iteration,
				generator.accumulator.String(),
			),
		),
	)
}

func (generator *Generator) toolcalls() {
	blocks := utils.ExtractJSONBlocks(generator.accumulator.String())
	for _, block := range blocks {
		if toolname, ok := block["tool"].(string); ok {
			if args, ok := block["args"].(map[string]any); ok {
				generator.toolcall(toolname, args)
			}
		}
	}
}

func (generator *Generator) toolcall(toolname string, args map[string]any) {
	generator.accumulator.Append(
		generator.updateStatus(
			generator.config.Toolset.Use(toolname, args),
		),
	)
}

func (generator *Generator) updateStatus(str string) string {
	switch str {
	case "break":
		generator.status = AgentStatusIdle
	default:
		generator.status = AgentStatusBusy
	}

	return str
}
