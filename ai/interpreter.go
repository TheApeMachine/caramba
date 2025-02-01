package ai

import (
	"strings"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/ai/tasks"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
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
func NewInterpreter(ctx *drknow.Context) *Interpreter {
	return &Interpreter{
		ctx:      ctx,
		commands: make([]Command, 0),
	}
}

func (interpreter *Interpreter) Execute() string {
	if len(interpreter.commands) == 0 {
		return ""
	}

	out := strings.Builder{}

	for _, command := range interpreter.commands {
		if command.Task == nil {
			continue
		}

		if command.Args == nil {
			command.Args = make(map[string]any)
		}

		answer := command.Task.Execute(interpreter.ctx, command.Args)
		out.WriteString(answer)
	}

	return out.String()
}

func (interpreter *Interpreter) Interpret() (*Interpreter, AgentState) {
	interpreter.commands = make([]Command, 0)
	agentState := AgentStateGenerating

	lastMessage := interpreter.ctx.LastMessage()

	if lastMessage == nil || lastMessage.Role != provider.RoleAssistant {
		errnie.Warn("last message is not an assistant message")
		return interpreter, agentState
	}

	blocks := utils.ExtractJSONBlocks(lastMessage.Content)

	for _, block := range blocks {
		if tool, ok := block["tool"].(string); ok {
			cmd := append(interpreter.commands, Command{
				Task: taskMap[tool],
				Args: block,
			})

			switch tool {
			case "break":
				agentState = AgentStateDone
			case "terminal":
				agentState = AgentStateTerminal
			}

			interpreter.commands = cmd
		}
	}

	return interpreter, agentState
}
