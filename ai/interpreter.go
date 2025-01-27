package ai

import (
	"regexp"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/ai/tasks"
	"github.com/theapemachine/caramba/provider"
)

var taskMap = map[string]tasks.Task{
	"ignore":   tasks.NewIgnore(),
	"help":     tasks.NewHelp(),
	"break":    tasks.NewBreak(),
	"recall":   tasks.NewRecall(),
	"remember": tasks.NewRemember(),
	"bash":     tasks.NewBash(),
	"terminal": tasks.NewTerminal(),
	"web":      tasks.NewWeb(),
	"optimize": tasks.NewOptimize(),
}

type Command struct {
	Task tasks.Task
	Args map[string]any
}

/*
Interpreter is an object that extracts and interprets commands from unstructured text.
It maps any commands to handler methods.
*/
type Interpreter struct {
	ctx      *drknow.Context
	commands []Command
}

/*
NewInterpreter creates a new Interpreter.
*/
func NewInterpreter(
	ctx *drknow.Context,
) *Interpreter {
	return &Interpreter{
		ctx:      ctx,
		commands: make([]Command, 0),
	}
}

func (interpreter *Interpreter) Execute() tasks.Bridge {
	var bridge tasks.Bridge

	for _, command := range interpreter.commands {
		bridge = command.Task.Execute(interpreter.ctx, command.Args)
	}

	return bridge
}

func (interpreter *Interpreter) Interpret() (*Interpreter, AgentState) {
	// Clear previous commands
	interpreter.commands = make([]Command, 0)

	agentState := AgentStateGenerating

	// Get the last message from context
	messages := interpreter.ctx.Identity.Params.Thread.Messages
	if len(messages) == 0 {
		return interpreter, agentState
	}
	lastMsg := messages[len(messages)-1]

	// Only process assistant messages
	if lastMsg.Role != provider.RoleAssistant {
		return interpreter, AgentStateGenerating
	}

	// This regex matches both:
	// <<command>> and <<command param1="value1" param2=[value2]>>
	regexpattern := regexp.MustCompile(`<<(\w+)(?:\s+(\w+)\s*=\s*(?:"([^"]*)"|(\[[^\]]*\])))?>>`)
	matches := regexpattern.FindAllStringSubmatch(lastMsg.Content, -1)

	for _, match := range matches {
		command := strings.ToLower(match[1])
		args := make(map[string]any)

		// The full match is at index 0, command name at index 1
		// After that, every group of 3 elements represents: key, quoted value, array value
		for i := 2; i < len(match); i += 3 {
			if match[i] != "" { // If we have a parameter name
				key := match[i]
				// Value could be either quoted string or array
				value := match[i+1]
				if value == "" {
					value = match[i+2] // Use array value if quoted string is empty
				}
				if value != "" {
					args[key] = value
				}
			}
		}

		if _, ok := taskMap[command]; !ok {
			log.Warn("Unknown command", "command", command)
			continue
		}

		if command == "terminal" {
			agentState = AgentStateTerminal
		}

		interpreter.commands = append(interpreter.commands, Command{
			Task: taskMap[command],
			Args: args,
		})
	}

	return interpreter, agentState
}
